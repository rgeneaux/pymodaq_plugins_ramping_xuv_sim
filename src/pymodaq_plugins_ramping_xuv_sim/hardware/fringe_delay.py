"""Mixin turning a camera viewer plugin into a delay reader (0D) from the fringes seen by the camera.

Use it with the plugin of any camera of which ``emit_data(frame)`` is called with the new frame(s), for instance::

    class DAQ_0DViewer_MyCameraFringeDelay(FringeDelayMixin, DAQ_2DViewer_MyCamera):
        params = DAQ_2DViewer_MyCamera.params + FringeDelayMixin.fringe_params

The resulting plugin is configured as the camera plugin (serial number, exposure, hardware ROI...) with an extra
*Fringe analysis* group of parameters. It emits, for each frame:

* ``delay``: the delay in fs (0D) tracked beyond one optical cycle, NaN if the fringes are not visible enough
* ``visibility``: the fringe visibility (0D, not plotted)
* ``Camera image``, ``Fringes`` and ``Fourier transform``: diagnostics (2D, 1D, 1D) that can be hidden

The fringe region and the Fourier transform peak are selected graphically with the *ROI select* tool of the viewers
of the camera image and of the Fourier transform. See :class:`FringePhaseTracker` for the analysis.
"""
import time
from typing import List, Optional

import numpy as np

from pymodaq_data.data import DataToExport, Axis
from pymodaq_gui.parameter import Parameter
from pymodaq_utils.utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins

from pymodaq_plugins_ramping_xuv_sim.hardware.fringe_phase import FringePhaseTracker

try:
    from pymodaq_gui.plotting.items.roi import RoiInfo  # pymodaq > 5.1.x
except ImportError:  # pragma: no cover
    from pymodaq_gui.plotting.utils.plot_utils import RoiInfo

IMAGE, FRINGES, FFT = 'Camera image', 'Fringes', 'Fourier transform'


class FringeDelayMixin:
    """Add the fringe analysis to a camera plugin (see the module docstring)"""
    fringe_params = [
        {'title': 'Fringe analysis', 'name': 'fringe', 'type': 'group', 'children': [
            {'title': 'Delay (fs):', 'name': 'delay', 'type': 'float', 'value': 0., 'readonly': True,
             'tip': 'Last measured delay (NaN if the fringes are not visible)'},
            {'title': 'Visibility:', 'name': 'visibility', 'type': 'float', 'value': 0., 'readonly': True},
            {'title': 'Tracking:', 'name': 'tracking', 'type': 'led', 'value': False, 'readonly': True,
             'tip': 'On when the fringes are visible enough to follow the phase'},
            {'title': 'Wavelength (nm):', 'name': 'wavelength', 'type': 'float', 'value': 790., 'min': 1.,
             'tip': 'Central wavelength, defining the delay of one optical cycle'},
            {'title': 'Sign:', 'name': 'sign', 'type': 'list', 'limits': ['+', '-'],
             'tip': 'Sign of the delay with respect to the phase (+ or -)'},
            {'title': 'Set delay to zero', 'name': 'set_zero', 'type': 'bool_push', 'value': False},
            {'title': 'Reset tracking', 'name': 'reset', 'type': 'bool_push', 'value': False,
             'tip': 'Forget the tracked phase (starts a new unwrapping)'},
            {'title': 'Visibility threshold:', 'name': 'threshold', 'type': 'float', 'value': 0.05, 'min': 0.,
             'max': 1., 'tip': 'Below this fringe visibility the frame is rejected and the delay is NaN'},
            {'title': 'Timestamp latency (ms):', 'name': 'latency', 'type': 'float', 'value': 0.,
             'tip': 'Time between the end of the exposure and the reception of the frame. '
                    'The timestamp of the delay is the middle of the exposure.'},
            {'title': 'Fringe region', 'name': 'region', 'type': 'group', 'children': [
                {'title': 'Use viewer ROI:', 'name': 'use_viewer_roi', 'type': 'bool', 'value': True,
                 'tip': 'The ROI drawn (ROI select tool) in the camera image viewer defines the fringe region '
                        'and the one in the Fourier transform viewer defines the peak'},
                {'title': 'Fringes along:', 'name': 'along', 'type': 'list', 'limits': ['x', 'y']},
                {'title': 'x0:', 'name': 'x0', 'type': 'int', 'value': 0, 'min': 0},
                {'title': 'y0:', 'name': 'y0', 'type': 'int', 'value': 0, 'min': 0},
                {'title': 'Width (0: all):', 'name': 'width', 'type': 'int', 'value': 0, 'min': 0},
                {'title': 'Height (0: all):', 'name': 'height', 'type': 'int', 'value': 0, 'min': 0},
            ]},
            {'title': 'Fourier transform peak', 'name': 'peak', 'type': 'group', 'children': [
                {'title': 'Automatic:', 'name': 'auto', 'type': 'bool', 'value': True,
                 'tip': 'Use the strongest peak (+/- half width)'},
                {'title': 'Half width:', 'name': 'half_width', 'type': 'int', 'value': 1, 'min': 0},
                {'title': 'First index:', 'name': 'first', 'type': 'int', 'value': 0, 'min': 0},
                {'title': 'Last index:', 'name': 'last', 'type': 'int', 'value': 1, 'min': 0},
            ]},
            {'title': 'Display', 'name': 'display', 'type': 'group', 'children': [
                {'title': 'Camera image:', 'name': 'show_image', 'type': 'bool', 'value': True},
                {'title': 'Fringes:', 'name': 'show_fringes', 'type': 'bool', 'value': False},
                {'title': 'Fourier transform:', 'name': 'show_fft', 'type': 'bool', 'value': True},
            ]},
        ]},
    ]

    def ini_attributes(self):
        super().ini_attributes()
        self.tracker = FringePhaseTracker()
        self._plotted: List[str] = []  # names of the plotted data, in the order of the viewers
        self._last_status = 0.  # time.time() of the last update of the status parameters

    # ---- parameters --------------------------------------------------------------------------------------------
    def ini_detector(self, controller=None):
        ret = super().ini_detector(controller)
        self._update_tracker()
        return ret

    def _update_tracker(self):
        f = self.settings.child('fringe')
        self.tracker.wavelength = f['wavelength']
        self.tracker.sign = 1 if f['sign'] == '+' else -1
        self.tracker.visibility_threshold = f['threshold']
        peak = f.child('peak')
        self.tracker.half_width = peak['half_width']
        self.tracker.band = None if peak['auto'] else (peak['first'], peak['last'])
        for name in ('first', 'last'):
            peak.child(name).setOpts(enabled=not peak['auto'])

    def commit_settings(self, param: Parameter):
        if param.parent() is not None and self._is_fringe_param(param):
            if param.name() == 'set_zero':
                if param.value():
                    self.tracker.set_zero()
                    param.setValue(False)
            elif param.name() == 'reset':
                if param.value():
                    self.tracker.reset()
                    param.setValue(False)
            else:
                self._update_tracker()
        else:
            super().commit_settings(param)

    def _is_fringe_param(self, param: Parameter) -> bool:
        parent = param
        while parent is not None:
            if parent.name() == 'fringe':
                return True
            parent = parent.parent()
        return False

    # ---- graphical selection of the fringes region and of the peak ---------------------------------------------
    def roi_select(self, roi_info: RoiInfo, ind_viewer: int = 0):
        """Called when the ROI of the camera image or Fourier transform viewer is moved"""
        try:
            name = self._plotted[ind_viewer]
        except IndexError:
            name = None
        super().roi_select(roi_info, ind_viewer)  # the camera plugin may use it (hardware ROI)
        region = self.settings.child('fringe', 'region')
        if not region['use_viewer_roi']:
            return
        if name == IMAGE and len(roi_info.origin) == 2:
            x_axis, y_axis = getattr(self, 'x_axis', None), getattr(self, 'y_axis', None)
            (y0, x0), (h, w) = roi_info.origin, roi_info.size
            if x_axis is not None and y_axis is not None:  # from viewer coordinates to image indices
                x0, w = (x0 - x_axis.offset) / x_axis.scaling, w / x_axis.scaling
                y0, h = (y0 - y_axis.offset) / y_axis.scaling, h / y_axis.scaling
            region.child('x0').setValue(max(int(round(x0)), 0))
            region.child('y0').setValue(max(int(round(y0)), 0))
            region.child('width').setValue(int(round(w)))
            region.child('height').setValue(int(round(h)))
        elif name == FFT:
            peak = self.settings.child('fringe', 'peak')
            first, size = roi_info.origin[0], roi_info.size[0]
            peak.child('first').setValue(max(int(round(first)), 0))
            peak.child('last').setValue(max(int(round(first + size)), 0))
            peak.child('auto').setValue(False)

    # ---- data --------------------------------------------------------------------------------------------------
    def emit_data(self, frame: Optional[np.ndarray] = None):
        """Replaces the emission of the image by the camera plugin: analyse the new frame and emit the delay"""
        try:
            t_received = time.time()
            if frame is None:
                frame = self.controller.read_newest_image()
            if frame is None:
                return
            frame = np.asarray(frame)
            image = frame.reshape((-1,) + frame.shape[-2:]).mean(axis=0)  # average of the frames, if several
            self.dte_signal.emit(self.analyse_image(image, t_received))
            if hasattr(self, 'update_fps') and self.settings['timing_opts', 'fps_on']:
                self.update_fps()
        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))

    def analyse_image(self, image: np.ndarray, t_received: float) -> DataToExport:
        """Delay and diagnostics from one image

        Parameters
        ----------
        image: np.ndarray
            2D image
        t_received: float
            time.time() when the image has been received
        """
        f = self.settings.child('fringe')
        region = f.child('region')
        fringes = self.tracker.fringes_from_image(
            image, (region['x0'], region['y0'], region['width'], region['height']), region['along'])
        res = self.tracker.process(fringes)

        if t_received - self._last_status > 0.25:  # do not flood the GUI at the frame rate
            self._last_status = t_received
            f.child('delay').setValue(res.delay)
            f.child('visibility').setValue(float(res.visibility))
            f.child('tracking').setValue(res.valid)
            if f['peak', 'auto']:  # show the detected peak
                f.child('peak', 'first').setValue(res.band[0])
                f.child('peak', 'last').setValue(res.band[1])

        # the timestamp is the middle of the exposure
        try:
            exposure = self.settings['timing_opts', 'exposure_time'] / 1000
        except KeyError:
            exposure = 0.
        timestamp = t_received - f['latency'] / 1000 - exposure / 2

        delay = DataFromPlugins(name='delay', data=[np.array([res.delay])], dim='Data0D', labels=['delay'],
                                units='fs')
        delay.timestamp = timestamp
        visibility = DataFromPlugins(name='visibility', data=[np.array([res.visibility])], dim='Data0D',
                                     labels=['visibility'], do_plot=False)
        visibility.timestamp = timestamp
        data = [delay, visibility]
        show = f.child('display')
        if show['show_image']:
            axes = [ax for ax in (getattr(self, 'y_axis', None), getattr(self, 'x_axis', None)) if ax is not None]
            data.append(DataFromPlugins(name=IMAGE, data=[image], dim='Data2D', labels=['intensity'],
                                        axes=axes if len(axes) == 2 else []))
        if show['show_fringes']:
            data.append(DataFromPlugins(name=FRINGES, data=[res.fringes], dim='Data1D', labels=['fringes']))
        if show['show_fft']:
            data.append(DataFromPlugins(
                name=FFT, data=[res.spectrum], dim='Data1D', labels=['|FT|'],
                axes=[Axis('Fourier index', data=np.arange(len(res.spectrum)), index=0)]))
        self._plotted = [dwa.name for dwa in data if getattr(dwa, 'do_plot', True)]
        return DataToExport('FringeDelay', data=data)
