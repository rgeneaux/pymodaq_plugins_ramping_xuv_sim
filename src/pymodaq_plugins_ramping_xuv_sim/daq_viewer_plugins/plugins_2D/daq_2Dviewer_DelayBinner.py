from typing import TYPE_CHECKING

from pymodaq_data.data import DataToExport
from pymodaq_gui.parameter import Parameter

from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main

if TYPE_CHECKING:
    from pymodaq_plugins_ramping_xuv_sim.extensions.delay_binner import DelayBinner


class DAQ_2DViewer_DelayBinner(DAQ_Viewer_base):
    """ Viewer module created by the Delay Binner extension (not a hardware plugin).

    It gives, as a standard DAQ_Viewer, the dR/R map (delay bins x photon energy) and the number of spectra per bin
    built by the extension. Data are pushed whenever the extension updates its map; a snap gives the current map.
    """
    params = comon_parameters

    def ini_attributes(self):
        self.controller: 'DelayBinner' = None

    def commit_settings(self, param: Parameter):
        pass

    def ini_detector(self, controller=None):
        self.controller = controller
        if self.controller is not None:
            self.controller.dte_computed_signal.connect(self.grab_done)
        return "DelayBinner viewer initialized", self.controller is not None

    def close(self):
        if self.controller is not None:
            try:
                self.controller.dte_computed_signal.disconnect(self.grab_done)
            except (TypeError, RuntimeError):
                pass

    def grab_done(self, dte: DataToExport):
        self.dte_signal.emit(dte)

    def grab_data(self, Naverage=1, **kwargs):
        """Emit the current map"""
        dte = self.controller.current_dte()
        if dte is not None:
            self.dte_signal.emit(dte)

    def stop(self):
        return ''


if __name__ == '__main__':
    main(__file__)
