pymodaq_plugins_ramping_xuv_sim
###############################

Simulated attosecond transient-reflectivity experiment, and the tools to acquire it with an oscillating delay stage
and an optical measurement of the delay.

Instruments
===========

* ``DAQ_Move_PiezoDelay``: simulated piezo delay stage (fs, finite velocity, or triangular waveform).
* ``DAQ_1DViewer_XUVReflectivity``: simulated XUV spectrometer returning dR/R vs photon energy, built from the
  experimental trace ``resources/MgO_Yb_proc.npz`` at the (true) delay of the stage.
  The noise decreases as 1/sqrt(exposure time).

Delay readers (0D, fs)
----------------------

The delay is measured optically: a camera sees the fringes of the interferometer, the phase of the fringes is
extracted (Fourier transform peak) and tracked, accounting for the 2 pi jumps, to get the delay beyond one optical
cycle (see ``hardware/fringe_phase.py``).

* ``DAQ_0DViewer_Thorlabs_TSI_FringeDelay``: the Thorlabs TSI camera plugin (``pymodaq_plugins_thorlabs``) analysing
  its own frames. Configure the camera as usual then, in the *Fringe analysis* parameters, set the wavelength
  and check *Use viewer ROI*: draw (ROI select tool of the viewers) a ROI on the camera image to select the
  fringes region, and a ROI on the Fourier transform viewer to select the peak (or keep the automatic detection).
  *Set delay to zero* defines the zero delay, *Reset tracking* restarts the unwrapping. The delay is NaN when
  the fringe visibility is below the threshold. The delay starts within one cycle of the zero: an optical phase gives
  the delay modulo one cycle only.
* ``DAQ_0DViewer_FringeDelaySim`` with ``DAQ_2DViewer_FringeCameraSim``: same, with a simulated camera.
* ``DAQ_0DViewer_InterferometerDelay``: simplest simulated reader (true delay + noise).

Delay Binner extension
======================

Builds a 2D dR/R map (delay x photon energy) from a spectrometer and a delay reader running independently (data
are matched with their timestamps), averaging the spectra in pre-defined delay bins. *Create Viewer* adds a 2D
viewer module to the dashboard with a *snap* (acquisition during a given time) and a continuous grab.

Authors
=======

* Romain Geneaux (romain.geneaux@cea.fr)
