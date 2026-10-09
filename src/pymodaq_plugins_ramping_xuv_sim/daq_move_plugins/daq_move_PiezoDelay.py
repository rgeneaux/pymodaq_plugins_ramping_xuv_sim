from typing import Union, List, Dict

from pymodaq.control_modules.move_utility_classes import (DAQ_Move_base, comon_parameters_fun,
                                                          main, DataActuatorType, DataActuator)
from pymodaq.control_modules.daq_move_ui.utils import UiType

from pymodaq_utils.utils import ThreadCommand
from pymodaq_gui.parameter import Parameter

from pymodaq_plugins_ramping_xuv_sim.hardware.xuv_sim import XUVExperimentSim


class DAQ_Move_PiezoDelay(DAQ_Move_base):
    """ Simulated piezo delay stage of an attosecond transient-reflectivity experiment.

    The stage position is a pump-probe delay in fs. It moves at a finite velocity (``velocity`` parameter) and its
    position is read live, so it can be swept continuously (e.g. with the ramping extension). Its position
    is shared with the ``XUVReflectivity`` 1D viewer plugin, which returns the dR/R spectrum at this delay.

    Travel range: the delay range of the experimental data (-7 to 6.8 fs).

    Waveform mode: with *Oscillate* checked the piezo oscillates (triangular waveform) between *min* and *max* at
    *osc velocity*, independently of PyMoDAQ. Its reported position is then only nominal: the delay really seen by
    the sample is measured by the InterferometerDelay viewer. Any move command stops the oscillation.
    """
    is_multiaxes = False
    _axis_names: Union[List[str], Dict[str, int]] = ['Delay']
    _controller_units: Union[str, List[str]] = XUVExperimentSim.units
    _epsilon: Union[float, List[float]] = 0.02

    data_actuator_type = DataActuatorType.DataActuator
    ui_type = UiType.NONE
    has_encoder = True

    params = [
        {'title': 'Velocity (fs/s):', 'name': 'velocity', 'type': 'float', 'value': 20., 'min': 1e-3,
         'tip': 'Piezo velocity. A ramp must not request more than this.'},
        {'title': 'Waveform:', 'name': 'waveform', 'type': 'group', 'children': [
            {'title': 'Oscillate:', 'name': 'oscillate', 'type': 'bool', 'value': False,
             'tip': 'Apply a triangular waveform between min and max'},
            {'title': 'Min (fs):', 'name': 'osc_min', 'type': 'float', 'value': -7.},
            {'title': 'Max (fs):', 'name': 'osc_max', 'type': 'float', 'value': 6.8},
            {'title': 'Velocity (fs/s):', 'name': 'osc_velocity', 'type': 'float', 'value': 1., 'min': 1e-3},
        ]},
    ] + comon_parameters_fun(is_multiaxes, axis_names=_axis_names, epsilon=_epsilon)

    def ini_attributes(self):
        self.controller: XUVExperimentSim = None

    def get_actuator_value(self) -> DataActuator:
        """Current stage position (fs), computed from the elapsed time of the current move"""
        pos = DataActuator(data=self.controller.get_position(), units=self.axis_unit)
        return self.get_position_with_scaling(pos)

    def close(self):
        """Nothing to close: the simulated controller is shared"""
        if self.controller is not None:
            self.controller.stop()

    def commit_settings(self, param: Parameter):
        if param.name() == 'velocity':
            self.controller.velocity = param.value()
        elif param.name() in ('oscillate', 'osc_min', 'osc_max', 'osc_velocity'):
            wf = self.settings.child('waveform')
            if wf['oscillate']:
                self.controller.start_oscillation(wf['osc_min'], wf['osc_max'], wf['osc_velocity'])
            elif self.controller.oscillating:
                self.controller.stop()

    def ini_stage(self, controller=None):
        if self.is_master:
            self.controller = XUVExperimentSim.shared()
        else:
            self.controller = controller
        self.controller.velocity = self.settings['velocity']
        info = (f"Simulated piezo delay stage, range [{self.controller.min_position}, "
                f"{self.controller.max_position}] fs")
        return info, True

    def move_abs(self, value: DataActuator):
        value = self.check_bound(value)
        self.target_value = value
        value = self.set_position_with_scaling(value)
        self.controller.move_at(value.value(self.axis_unit))
        self._uncheck_oscillate()

    def move_rel(self, value: DataActuator):
        value = self.check_bound(self.current_value + value) - self.current_value
        self.target_value = value + self.current_value
        value = self.set_position_relative_with_scaling(value)
        self.controller.move_at(self.controller.get_position() + value.value(self.axis_unit))
        self._uncheck_oscillate()

    def move_home(self):
        """Go to zero delay"""
        self.move_abs(DataActuator(data=0., units=self.axis_unit))

    def _uncheck_oscillate(self):
        """A point to point move cancels the waveform"""
        self.settings.child('waveform', 'oscillate').setValue(False)

    def stop_motion(self):
        self.controller.stop()
        self._uncheck_oscillate()
        self.emit_status(ThreadCommand('Update_Status', ['Piezo stopped']))


if __name__ == '__main__':
    main(__file__)
