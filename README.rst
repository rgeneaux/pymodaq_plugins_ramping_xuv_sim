pymodaq_plugins_ramping_xuv_sim
###############################

Simulated attosecond transient-reflectivity experiment, used to test the PyMoDAQ *ramping* extension.

Instruments
===========

* ``DAQ_Move_PiezoDelay``: simulated piezo delay stage (units: fs, finite velocity).
* ``DAQ_1DViewer_XUVReflectivity``: simulated XUV spectrometer returning dR/R vs photon energy,
  built from the experimental trace ``resources/MgO_Yb_proc.npz`` at the current stage delay.
  The noise decreases as 1/sqrt(exposure time).

Authors
=======

* Romain Geneaux (romain.geneaux@cea.fr)
