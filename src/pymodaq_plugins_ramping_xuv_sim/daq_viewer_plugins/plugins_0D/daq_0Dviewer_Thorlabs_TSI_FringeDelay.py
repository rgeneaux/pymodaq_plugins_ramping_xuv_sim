from pymodaq.control_modules.viewer_utility_classes import main

from pymodaq_plugins_ramping_xuv_sim.hardware.fringe_delay import FringeDelayMixin
from pymodaq_plugins_thorlabs.daq_viewer_plugins.plugins_2D.daq_2Dviewer_Thorlabs_TSI import (
    DAQ_2DViewer_Thorlabs_TSI)


class DAQ_0DViewer_Thorlabs_TSI_FringeDelay(FringeDelayMixin, DAQ_2DViewer_Thorlabs_TSI):
    """ Delay reader from the fringes seen by a Thorlabs TSI scientific camera (interferometer delay).

    It is the DAQ_2DViewer_Thorlabs_TSI plugin (same parameters: serial number, exposure, hardware ROI/binning...)
    which analyses the fringes of each frame instead of emitting the image: the phase of the fringes is extracted and
    tracked (2 pi jumps accounted for) to get the delay in fs, beyond one optical cycle.

    See :class:`FringeDelayMixin` for the description of the data and of the parameters. Requires the
    pymodaq_plugins_thorlabs package.
    """
    params = DAQ_2DViewer_Thorlabs_TSI.params + FringeDelayMixin.fringe_params


if __name__ == '__main__':
    main(__file__, init=False)
