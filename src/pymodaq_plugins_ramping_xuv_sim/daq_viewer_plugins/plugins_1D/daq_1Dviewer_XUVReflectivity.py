import time

import numpy as np

from pymodaq_utils.utils import ThreadCommand
from pymodaq_data.data import DataToExport, Axis
from pymodaq_gui.parameter import Parameter

from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.data import DataFromPlugins

from pymodaq_plugins_ramping_xuv_sim.hardware.xuv_sim import XUVExperimentSim


class DAQ_1DViewer_XUVReflectivity(DAQ_Viewer_base):
    """ Simulated XUV spectrometer measuring the transient reflectivity dR/R versus photon energy.

    The spectrum is taken from an experimental (delay, energy) map, at the delay of the simulated
    ``PiezoDelay`` actuator (linear interpolation, averaged over the delays travelled during the exposure
    if the stage is moving).

    Each grab lasts ``exposure`` and the noise rms is ``noise_1s / sqrt(exposure [s] * Naverage)``: the longer
    the integration, the lower the noise.
    """
    params = comon_parameters + [
        {'title': 'Exposure (ms):', 'name': 'exposure', 'type': 'float', 'value': 100., 'min': 1., 'suffix': 'ms'},
        {'title': 'Noise for 1 s exposure:', 'name': 'noise_1s', 'type': 'float', 'value': 2e-3, 'min': 0.,
         'tip': 'rms noise on dR/R for a 1 s integration; scales as 1/sqrt(exposure)'},
        {'title': 'Noise (rms):', 'name': 'noise', 'type': 'float', 'value': 0., 'readonly': True,
         'tip': 'rms noise at the current exposure and averaging'},
        {'title': 'True delay (sim only, fs):', 'name': 'delay', 'type': 'float', 'value': 0., 'readonly': True},
    ]

    def ini_attributes(self):
        self.controller: XUVExperimentSim = None
        self.x_axis = None

    def _noise_rms(self, n_average: int = 1) -> float:
        exposure = self.settings['exposure'] / 1000
        return self.settings['noise_1s'] / np.sqrt(exposure * n_average)

    def commit_settings(self, param: Parameter):
        if param.name() in ('exposure', 'noise_1s'):
            self.settings.child('noise').setValue(self._noise_rms())

    def ini_detector(self, controller=None):
        if self.is_master:
            self.controller = XUVExperimentSim.shared()
        else:
            self.controller = controller

        self.x_axis = Axis(data=self.controller.energy, label='Photon energy', units='eV', index=0)
        self.settings.child('noise').setValue(self._noise_rms())
        self.dte_signal_temp.emit(DataToExport(
            name='XUVReflectivity',
            data=[DataFromPlugins(name='dR/R', data=[np.zeros_like(self.controller.energy)], dim='Data1D',
                                  labels=['dR/R'], axes=[self.x_axis])]))
        return "Simulated XUV spectrometer", True

    def close(self):
        pass

    def grab_data(self, Naverage=1, **kwargs):
        """Integrate for the exposure time and emit the noisy dR/R spectrum"""
        exposure = self.settings['exposure'] / 1000
        spectrum, delay = self.controller.measure(exposure, self.settings['noise_1s'], n_average=Naverage)
        t_mid = time.time() + exposure / 2
        time.sleep(exposure)  # the acquisition takes the exposure time, as in a real spectrometer
        self.settings.child('delay').setValue(delay)
        self.settings.child('noise').setValue(self._noise_rms(Naverage))
        dwa = DataFromPlugins(name='dR/R', data=[spectrum], dim='Data1D', labels=['dR/R'], axes=[self.x_axis])
        dwa.timestamp = t_mid  # middle of the integration, used to match this spectrum with the measured delay
        self.dte_signal.emit(DataToExport('XUVReflectivity', data=[dwa]))

    def stop(self):
        return ''


if __name__ == '__main__':
    main(__file__)
