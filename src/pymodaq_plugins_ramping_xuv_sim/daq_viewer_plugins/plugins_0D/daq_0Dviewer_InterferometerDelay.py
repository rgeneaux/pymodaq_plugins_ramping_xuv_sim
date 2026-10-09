import time

import numpy as np

from pymodaq_data.data import DataToExport
from pymodaq_gui.parameter import Parameter

from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.data import DataFromPlugins

from pymodaq_plugins_ramping_xuv_sim.hardware.xuv_sim import XUVExperimentSim


class DAQ_0DViewer_InterferometerDelay(DAQ_Viewer_base):
    """ Simulated optical measurement of the pump-probe delay (phase of the interferometer fringes).

    Returns the *true* delay (fs) seen by the sample, which differs from the position claimed by the piezo
    (non-linearity, drift), plus a gaussian measurement noise. It runs at its own acquisition rate, independent of
    the spectrometer. The timestamp of each point is the time of the measurement.
    """
    params = comon_parameters + [
        {'title': 'Rate (Hz):', 'name': 'rate', 'type': 'float', 'value': 200., 'min': 1., 'max': 5000.,
         'tip': 'Acquisition rate of the delay measurement'},
        {'title': 'Noise (fs rms):', 'name': 'noise', 'type': 'float', 'value': 0.01, 'min': 0.},
    ]

    def ini_attributes(self):
        self.controller: XUVExperimentSim = None

    def commit_settings(self, param: Parameter):
        pass

    def ini_detector(self, controller=None):
        self.controller = XUVExperimentSim.shared() if self.is_master else controller
        self.dte_signal_temp.emit(DataToExport(
            name='InterferometerDelay',
            data=[DataFromPlugins(name='delay', data=[np.array([0.])], dim='Data0D', labels=['delay'],
                                  units='fs')]))
        return "Simulated interferometer", True

    def close(self):
        pass

    def grab_data(self, Naverage=1, **kwargs):
        """Wait for the next sample (1/rate) and emit the measured delay with its timestamp"""
        time.sleep(1 / self.settings['rate'])
        t, delay = self.controller.measure_delay(self.settings['noise'])
        dwa = DataFromPlugins(name='delay', data=[np.array([delay])], dim='Data0D', labels=['delay'], units='fs')
        dwa.timestamp = t
        self.dte_signal.emit(DataToExport('InterferometerDelay', data=[dwa]))

    def stop(self):
        return ''


if __name__ == '__main__':
    main(__file__)
