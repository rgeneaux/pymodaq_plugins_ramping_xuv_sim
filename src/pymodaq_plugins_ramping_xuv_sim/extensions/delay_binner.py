"""Extension building a dR/R (delay x photon energy) map from an oscillating delay stage.

The piezo stage oscillates slowly between two positions. The real delay is measured optically (fringe phase) by one
detector (0D, fs) at its own rate, the dR/R spectra are measured by another detector (1D) at its own rate. Both
are continuously grabbed, matched with their timestamps, and the spectra are averaged in pre-defined delay bins.
The map is displayed in the extension and can also be exposed as a new DAQ_Viewer (2D) module of the dashboard.
"""
from typing import Optional, TYPE_CHECKING

import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq.control_modules.enums import DAQTypesEnum
from pymodaq.control_modules.utils import ControllerAndThread, QThreadProxy
from pymodaq.utils.custom_ext import CustomExt
from pymodaq.utils.managers.modules import ModuleType
from pymodaq.utils.managers.modules.loader import PluginInfo
from pymodaq_data.data import DataToExport, DataCalculated, Axis
from pymodaq_gui import utils as gutils
from pymodaq_gui.plotting.data_viewers.viewer import ViewerDispatcher
from pymodaq_utils.logger import set_logger, get_module_name

from pymodaq.control_modules.instruments import DET_TYPES
from pymodaq_plugins_ramping_xuv_sim.binning import DelayBinner as Binner, to_uniform_axis

if TYPE_CHECKING:
    from pymodaq.control_modules.daq_viewer import DAQ_Viewer

logger = set_logger(get_module_name(__file__))

EXTENSION_NAME = 'Delay Binner'  # the name that will be displayed in the extension list in the dashboard
CLASS_NAME = 'DelayBinner'  # the name of the class defined below

DISPLAY_PERIOD_MS = 200  # the map is refreshed at most at this period

VIEWER_PLUGIN = 'DelayBinner'  # name of the DAQ_2DViewer_DelayBinner plugin, defined in this extensions folder


def register_viewer_plugin():
    """Make the DAQ_2DViewer_DelayBinner plugin known to PyMoDAQ for the lifetime of the extension

    The plugin is only meant to be created by this extension: it is not in the plugins_2D folder (which would
    list it in the detectors available in any DAQ_Viewer or preset, where it can't work) but is added here to
    the list of the 2D plugins that PyMoDAQ uses to find the class of a module to create.
    """
    import importlib
    parent = importlib.import_module(__package__)
    importlib.import_module(f'{__package__}.daq_2Dviewer_{VIEWER_PLUGIN}')
    if not any(plugin['name'] == VIEWER_PLUGIN for plugin in DET_TYPES['DAQ2D']):
        DET_TYPES['DAQ2D'].append({'name': VIEWER_PLUGIN, 'module': parent, 'parent_module': parent,
                                   'type': 'daq_2Dviewer'})


def unregister_viewer_plugin():
    DET_TYPES['DAQ2D'][:] = [plugin for plugin in DET_TYPES['DAQ2D'] if plugin['name'] != VIEWER_PLUGIN]


class DelayBinner(CustomExt):
    """Bin the spectra of a 1D detector according to the delay measured by a 0D detector"""
    settings_name = 'DelayBinnerSettings'
    params = [
        {'title': 'Detectors', 'name': 'detectors', 'type': 'group', 'children': [
            {'title': 'dR/R spectra (1D):', 'name': 'spectrum', 'type': 'list', 'limits': []},
            {'title': 'Measured delay (0D, fs):', 'name': 'delay', 'type': 'list', 'limits': []},
        ]},
        {'title': 'Binning', 'name': 'binning', 'type': 'group', 'children': [
            {'title': 'Delay min (fs):', 'name': 'delay_min', 'type': 'float', 'value': -7.},
            {'title': 'Delay max (fs):', 'name': 'delay_max', 'type': 'float', 'value': 7.},
            {'title': 'Number of bins:', 'name': 'n_bins', 'type': 'int', 'value': 70, 'min': 1},
            {'title': 'Bin width (fs):', 'name': 'bin_width', 'type': 'float', 'value': 0.2, 'readonly': True},
            {'title': 'Spectrum integration (ms):', 'name': 'window', 'type': 'float', 'value': 100., 'min': 0.,
             'tip': 'Duration over which the measured delays are averaged for each spectrum '
                    '(should be the exposure of the spectrometer)'},
        ]},
        {'title': 'Status', 'name': 'status', 'type': 'group', 'children': [
            {'title': 'Spectra in map:', 'name': 'n_used', 'type': 'int', 'value': 0, 'readonly': True},
            {'title': 'Outside delay range:', 'name': 'n_out', 'type': 'int', 'value': 0, 'readonly': True},
            {'title': 'Without delay data:', 'name': 'n_no_delay', 'type': 'int', 'value': 0, 'readonly': True},
            {'title': 'Empty bins:', 'name': 'n_empty', 'type': 'int', 'value': 0, 'readonly': True},
        ]},
    ]

    dte_computed_signal = QtCore.Signal(DataToExport)  # the current map and counts, emitted periodically
    acquisition_done_signal = QtCore.Signal(DataToExport)  # the final map of a timed acquisition (viewer snap)
    request_signal = QtCore.Signal(str, float)  # ('start', duration in s, <= 0 for infinite) or ('stop', 0) from the viewer module

    def __init__(self, parent: gutils.DockArea, dashboard):
        register_viewer_plugin()
        self.binner: Optional[Binner] = None
        self._running = False
        self._timed = False
        self._dirty = False
        self._energy: Optional[np.ndarray] = None
        self._spectrum_det: Optional['DAQ_Viewer'] = None
        self._delay_det: Optional['DAQ_Viewer'] = None
        super().__init__(parent, dashboard, add_toolbar_break=False)

        self.binner = self._new_binner()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(DISPLAY_PERIOD_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._duration_timer = QtCore.QTimer(self)
        self._duration_timer.setSingleShot(True)
        self._duration_timer.timeout.connect(self.stop)

        self.setup_ui()

    # ---- UI ----------------------------------------------------------------------------------------------------
    def setup_docks_and_widgets(self):
        self.docks['settings'] = gutils.Dock('Settings')
        self.dockarea.addDock(self.docks['settings'])
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.docks['settings'].addWidget(splitter)
        splitter.addWidget(self.settings_tree)

        self.docks['map'] = gutils.Dock('dR/R map')
        self.dockarea.addDock(self.docks['map'], 'right')
        self.area_map = gutils.DockArea()
        self.docks['map'].addWidget(self.area_map)
        self.map_viewer = ViewerDispatcher(self.area_map)

    def setup_actions(self):
        self.add_action('start', 'Start', 'run2', 'Grab continuously both detectors and fill the map',
                        checkable=True, checked=False, auto_toolbar=True)
        self.add_action('reset', 'Reset map', 'reset', 'Empty the map', auto_toolbar=True)
        self.add_action('create_viewer', 'Create Viewer', 'Add_Step',
                        tip='Create a 2D DAQ_Viewer module of the dashboard giving the map')

    def setup_menus_and_toolbars(self, menubar: QtWidgets.QMenuBar = None):
        self.create_dashboard_toolbar(add_break=False)

    def connect_things(self):
        self.connect_action('start', self.start_acquisition)
        self.connect_action('reset', self.reset_map)
        self.connect_action('create_viewer', self.create_viewer)
        self.dte_computed_signal.connect(self.map_viewer.show_data)
        self.acquisition_done_signal.connect(self.map_viewer.show_data)
        self.request_signal.connect(self._on_request)

    def do_things_after_experiment_set(self, experiment_name: str, show_dashboard: bool = None):
        super().do_things_after_experiment_set(experiment_name, show_dashboard)
        self.update_detector_lists()

    def update_detector_lists(self):
        names = [n for n in self.modules_manager.detectors_name if n != 'DelayBinner']
        for child in ('spectrum', 'delay'):
            param = self.settings.child('detectors', child)
            param.setLimits(names)
        # guess: a detector with 'delay'/'interfero' in its name is the delay measurement
        for name in names:
            if 'delay' in name.lower() or 'interfero' in name.lower():
                self.settings.child('detectors', 'delay').setValue(name)
            elif 'xuv' in name.lower() or 'reflectivity' in name.lower():
                self.settings.child('detectors', 'spectrum').setValue(name)

    def value_changed(self, param):
        if param.name() in ('delay_min', 'delay_max', 'n_bins', 'window'):
            self._update_bin_width()
            if self.binner is not None:
                self.reset_map()

    def _update_bin_width(self):
        b = self.settings.child('binning')
        b.child('bin_width').setValue((b['delay_max'] - b['delay_min']) / b['n_bins'])

    def _new_binner(self) -> Binner:
        b = self.settings.child('binning')
        return Binner(b['delay_min'], b['delay_max'], b['n_bins'], b['window'] / 1000)

    # ---- acquisition -------------------------------------------------------------------------------------------
    # ---- acquisition -------------------------------------------------------------------------------------------
    def start_acquisition(self, start: bool):
        """Slot of the Start toolbar action: acquire for an infinite time, adding to the current map"""
        if start:
            self._start(duration=None, reset=False)
        else:
            self.stop()

    def _on_request(self, command: str, duration: float):
        """Slot of the requests of the viewer module (emitted from its thread):
        'start' with a duration in s (<= 0 for an infinite acquisition), or 'stop'"""
        if command == 'start':
            self._start(duration=duration if duration > 0 else None, reset=True)
        elif command == 'stop' and self._running:
            self.stop()

    def _set_start_checked(self, checked: bool):
        action = self.get_action('start')
        action.blockSignals(True)
        action.setChecked(checked)
        action.blockSignals(False)

    def _start(self, duration: Optional[float], reset: bool):
        """Grab both detectors continuously, for ``duration`` seconds or until stopped

        Parameters
        ----------
        duration: float or None
            None to acquire until :meth:`stop` is called. Otherwise the acquisition stops by itself and
            ``acquisition_done_signal`` is emitted with the final map.
        reset: bool
            Empty the map before starting
        """
        if self._running:
            self.stop()
        spectrum_name, delay_name = self.settings['detectors', 'spectrum'], self.settings['detectors', 'delay']
        self._spectrum_det = self.modules_manager.get_mod_from_name(spectrum_name, ModuleType.Detector)
        self._delay_det = self.modules_manager.get_mod_from_name(delay_name, ModuleType.Detector)
        if self._spectrum_det is None or self._delay_det is None or self._spectrum_det is self._delay_det:
            logger.error('Select two different detectors (spectra and delay)')
            self._set_start_checked(False)
            return
        if reset:
            self.reset_map()
        self._spectrum_det.grab_done_signal.connect(self._on_spectrum)
        self._delay_det.grab_done_signal.connect(self._on_delay)
        self._running = True
        self._timed = duration is not None
        self._set_start_checked(True)
        self.settings.child('detectors').setOpts(enabled=False)
        self._delay_det.grab_data(True)
        self._spectrum_det.grab_data(True)
        if duration is not None:
            self._duration_timer.start(int(duration * 1000))

    def stop(self):
        """Stop the detectors (the map is kept). After a timed acquisition, the map is sent with
        ``acquisition_done_signal``"""
        self._duration_timer.stop()
        was_timed = self._running and self._timed
        self._running = False
        self._timed = False
        self._timed = False
        for det, slot in ((self._spectrum_det, self._on_spectrum), (self._delay_det, self._on_delay)):
            if det is not None:
                det.grab_data(False)
                try:
                    det.grab_done_signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        self.settings.child('detectors').setOpts(enabled=True)
        self._set_start_checked(False)
        if was_timed:
            self._refresh(force=True)
            dte = self.current_dte()
            if dte is not None:
                self.acquisition_done_signal.emit(dte)

    def _on_delay(self, dte: DataToExport):
        for dwa in dte.get_data_from_dim('Data0D'):
            self.binner.add_delay(dwa.timestamp, float(np.ravel(dwa[0])[0]))
            self._dirty = True
            return

    def _on_spectrum(self, dte: DataToExport):
        for dwa in dte.get_data_from_dim('Data1D'):
            energy = dwa.axes[0].get_data() if dwa.axes else None
            self.binner.add_spectrum(dwa.timestamp, np.ravel(dwa[0]), energy)
            self._energy = self.binner.energy
            self._dirty = True
            return

    def reset_map(self):
        b = self.settings.child('binning')
        self.binner = self._new_binner() if self.binner is None else self.binner
        self.binner.window = b['window'] / 1000
        self.binner.reset(b['delay_min'], b['delay_max'], b['n_bins'])
        self._dirty = True

    # ---- results -----------------------------------------------------------------------------------------------
    def current_dte(self) -> Optional[DataToExport]:
        """The map (delay x energy, empty bins set to 0) and the number of spectra per bin"""
        map_ = self.binner.map
        if map_ is None:
            return None
        map_ = np.nan_to_num(map_)
        axes = [Axis('Delay', 'fs', data=self.binner.centers, index=0)]
        energy = self.binner.energy
        if energy is not None and len(energy) == map_.shape[1]:
            # the energy axis of a spectrometer is not linear but the 2D viewers need linear axes
            map_, energy = to_uniform_axis(map_, energy)
            axes.append(Axis('Photon energy', 'eV', data=energy, index=1))
        return DataToExport('DelayBinner', data=[
            DataCalculated('dR/R map', data=[map_], labels=['dR/R'], axes=axes),
            DataCalculated('Spectra per bin', data=[self.binner.counts.astype(float)], labels=['counts'],
                           axes=[Axis('Delay', 'fs', data=self.binner.centers, index=0)]),
        ])

    def _refresh(self, force: bool = False):
        if not (self._dirty or force) or self.binner is None:
            return
        self._dirty = False
        st = self.settings.child('status')
        st.child('n_used').setValue(self.binner.n_used)
        st.child('n_out').setValue(self.binner.n_out_of_range)
        st.child('n_no_delay').setValue(self.binner.n_no_delay)
        st.child('n_empty').setValue(int(np.sum(self.binner.counts == 0)))
        dte = self.current_dte()
        if dte is not None:
            self.dte_computed_signal.emit(dte)

    # ---- viewer module -----------------------------------------------------------------------------------------
    def create_viewer(self):
        """Create a 2D DAQ_Viewer module (plugin DAQ_2DViewer_DelayBinner) outputting the map"""
        id = self.dashboard.modules_manager.get_random_id()
        plugin = PluginInfo(
            id=id,
            name='DelayBinner',
            class_name='DelayBinner',
            type=ModuleType.Detector,
            daq_type=DAQTypesEnum.DAQ2D,
            controller=ControllerAndThread(name='DelayBinner',
                                           thread=QThreadProxy(thread=self.thread()),
                                           controller=self,
                                           is_master=False,
                                           id=id))
        self.dashboard.module_creator.add_det_from_extension(modules=[plugin], callback=self._on_viewer_added)

    def _on_viewer_added(self, modules: list['DAQ_Viewer']):
        self.dashboard.module_creator.forget_callback(self._on_viewer_added)
        self.set_action_enabled('create_viewer', False)
        self.update_detector_lists()

    def _quit_fun(self) -> bool:
        self.stop()
        self._timer.stop()
        self.dashboard.remove_modules([VIEWER_PLUGIN])
        unregister_viewer_plugin()
        return True


def main():
    import sys
    from pymodaq_gui.qt_utils import mkQApp
    from pymodaq.dashboard import load_dashboard_with_arguments
    from pymodaq.utils.gui_utils.loader_utils import create_extension

    app = mkQApp('Delay Binner')
    win, dashboard, _ = load_dashboard_with_arguments(show_dashboard=False, load_extension=False)
    win.mainwindow.setVisible(False)
    win_ext, ext = create_extension(dashboard, DelayBinner, show_extension=True)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
