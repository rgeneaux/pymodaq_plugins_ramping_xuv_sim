import time

import numpy as np
from qtpy import QtCore

from pymodaq_data.data import DataToExport, Axis
from pymodaq_gui.parameter import Parameter
from pymodaq_utils.utils import ThreadCommand

from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.data import DataFromPlugins

from pymodaq_plugins_ramping_xuv_sim.hardware.xuv_sim import XUVExperimentSim
from pymodaq_plugins_ramping_xuv_sim.hardware.fringe_phase import C


class DAQ_2DViewer_FringeCameraSim(DAQ_Viewer_base):
    """ Simulated camera seeing the fringes of the interferometer of the simulated experiment.

    The fringes move with the true delay of the simulated piezo stage (phase = 2 pi c delay / wavelength), they are
    modulated along x with a (non integer) period, have a gaussian beam profile and are averaged during the exposure
    (a fast moving delay washes them out). It has the same structure (``timing_opts`` parameters, frames emitted by
    ``emit_data``) as the pylablib camera plugins so that it can be used to test ``FringeDelayMixin``.
    """
    live_mode_available = True

    params = comon_parameters + [
        {'title': 'Image size', 'name': 'size', 'type': 'group', 'children': [
            {'title': 'Width:', 'name': 'width', 'type': 'int', 'value': 160, 'min': 16},
            {'title': 'Height:', 'name': 'height', 'type': 'int', 'value': 100, 'min': 4},
        ]},
        {'title': 'Fringes', 'name': 'fringes', 'type': 'group', 'children': [
            {'title': 'Wavelength (nm):', 'name': 'wavelength', 'type': 'float', 'value': 790.},
            {'title': 'Period (pixels):', 'name': 'period', 'type': 'float', 'value': 12.3},
            {'title': 'Contrast:', 'name': 'contrast', 'type': 'float', 'value': 0.6, 'min': 0., 'max': 1.},
            {'title': 'Phase offset (rad):', 'name': 'phase0', 'type': 'float', 'value': 0.7},
            {'title': 'Peak signal (counts):', 'name': 'peak', 'type': 'float', 'value': 3000.},
            {'title': 'Read noise (counts):', 'name': 'read_noise', 'type': 'float', 'value': 5.},
        ]},
        {'title': 'Timing', 'name': 'timing_opts', 'type': 'group', 'children': [
            {'title': 'Exposure Time (ms)', 'name': 'exposure_time', 'type': 'int', 'value': 2, 'min': 1},
            {'title': 'Frame rate (Hz)', 'name': 'frame_rate', 'type': 'float', 'value': 100., 'min': 1.},
            {'title': 'Compute FPS', 'name': 'fps_on', 'type': 'bool', 'value': False},
        ]},
    ]

    def ini_attributes(self):
        self.controller: XUVExperimentSim = None
        self.x_axis: Axis = None
        self.y_axis: Axis = None
        self._timer: QtCore.QTimer = None
        self._rng = np.random.default_rng()

    def commit_settings(self, param: Parameter):
        if param.name() == 'frame_rate' and self._timer is not None:
            self._timer.setInterval(int(1000 / param.value()))
        elif param.name() in ('width', 'height'):
            self._compute_axes()

    def _compute_axes(self):
        self.x_axis = Axis('x_axis', offset=0, scaling=1, size=self.settings['size', 'width'], index=1)
        self.y_axis = Axis('y_axis', offset=0, scaling=1, size=self.settings['size', 'height'], index=0)

    def ini_detector(self, controller=None):
        self.controller = XUVExperimentSim.shared() if self.is_master else controller
        self._compute_axes()
        self._timer = QtCore.QTimer(self)  # created in the thread of the plugin
        self._timer.setInterval(int(1000 / self.settings['timing_opts', 'frame_rate']))
        self._timer.timeout.connect(self._acquire_frame)
        self.dte_signal_temp.emit(DataToExport('FringeCamera', data=[DataFromPlugins(
            name='Camera', data=[np.zeros((self.settings['size', 'height'], self.settings['size', 'width']))],
            dim='Data2D', labels=['intensity'], axes=[self.y_axis, self.x_axis])]))
        return "Simulated fringe camera", True

    def close(self):
        if self._timer is not None:
            self._timer.stop()

    def grab_data(self, Naverage=1, live=False, **kwargs):
        """Live: frames at the frame rate until stopped. Snap: one frame"""
        if live:
            self._timer.start()
        else:
            self._acquire_frame()

    def stop(self):
        if self._timer is not None:
            self._timer.stop()
        return ''

    def make_frame(self) -> np.ndarray:
        """Image of the fringes at the delay(s) of the exposure ending now"""
        exposure = self.settings['timing_opts', 'exposure_time'] / 1000
        f = self.settings.child('fringes')
        t_end = time.time()
        times = t_end - exposure * np.linspace(1, 0, 5)
        phases = np.array([2 * np.pi * self.controller.true_delay_at(t) * 1e-15 * C / (f['wavelength'] * 1e-9)
                           for t in times]) + f['phase0']
        width, height = self.settings['size', 'width'], self.settings['size', 'height']
        x, y = np.arange(width), np.arange(height)[:, None]
        # the carrier is the same for all the sub-exposures, the average washes the fringes out if the phase moves
        carrier = np.mean([np.cos(2 * np.pi * x / f['period'] + p) for p in phases], axis=0)
        profile = np.exp(-((x - width / 2) ** 2 / (2 * (width / 4) ** 2) + (y - height / 2) ** 2 / (2 * (height / 3) ** 2)))
        signal = f['peak'] * profile * (1 + f['contrast'] * carrier) / (1 + f['contrast'])
        image = self._rng.poisson(np.maximum(signal, 0)) + self._rng.normal(0, f['read_noise'], signal.shape)
        return np.clip(image, 0, 4095).astype(np.float32)

    def _acquire_frame(self):
        self.emit_data(self.make_frame()[None, None])  # same layout as the pylablib cameras: (average, frame, y, x)

    def emit_data(self, frame: np.ndarray = None):
        """Emit the image of a frame (reimplemented by FringeDelayMixin to emit a delay instead)"""
        image = np.asarray(frame).reshape((-1,) + frame.shape[-2:]).mean(axis=0)
        self.dte_signal.emit(DataToExport('FringeCamera', data=[DataFromPlugins(
            name='Camera', data=[image], dim='Data2D', labels=['intensity'], axes=[self.y_axis, self.x_axis])]))


if __name__ == '__main__':
    main(__file__)
