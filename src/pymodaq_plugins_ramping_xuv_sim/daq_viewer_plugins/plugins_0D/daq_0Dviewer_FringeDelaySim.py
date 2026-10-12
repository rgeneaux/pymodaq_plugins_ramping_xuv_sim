from pymodaq.control_modules.viewer_utility_classes import main

from pymodaq_plugins_ramping_xuv_sim.hardware.fringe_delay import FringeDelayMixin
from pymodaq_plugins_ramping_xuv_sim.daq_viewer_plugins.plugins_2D.daq_2Dviewer_FringeCameraSim import (
    DAQ_2DViewer_FringeCameraSim)


class DAQ_0DViewer_FringeDelaySim(FringeDelayMixin, DAQ_2DViewer_FringeCameraSim):
    """ Delay reader from the fringes of the simulated camera (same as DAQ_0DViewer_Thorlabs_TSI_FringeDelay with a
    simulated camera). The phase of the fringes is extracted and tracked to get the delay in fs, beyond one optical
    cycle. See :class:`FringeDelayMixin` for the description of the data and of the parameters.
    """
    params = DAQ_2DViewer_FringeCameraSim.params + FringeDelayMixin.fringe_params


if __name__ == '__main__':
    main(__file__)
