# Turning a camera plugin into a fringe delay reader

`FringeDelayMixin` ([hardware/fringe_delay.py](../src/pymodaq_plugins_ramping_xuv_sim/hardware/fringe_delay.py))
adds the fringe analysis to an existing camera viewer plugin. The resulting plugin is a **0D viewer** that is
configured like the camera (serial number, exposure, hardware ROI...) and outputs the interferometer **delay in fs**
instead of the image. The analysis itself (Fourier peak phase, unwrapping) is in
[hardware/fringe_phase.py](../src/pymodaq_plugins_ramping_xuv_sim/hardware/fringe_phase.py).

Reference implementations to copy from:

* `daq_0Dviewer_FringeDelaySim.py` on top of `daq_2Dviewer_FringeCameraSim.py` (simulated camera, 15 lines)
* `daq_0Dviewer_Thorlabs_TSI_FringeDelay.py` on top of the Thorlabs TSI plugin

---

## 1. Check that your camera plugin is compatible

The mixin replaces the method **`emit_data(self, frame)`**, the one where the camera plugin emits a new frame.
Open the camera plugin and check:

| # | What the mixin needs | Where it is used | If your plugin differs |
|---|---|---|---|
| 1 | The plugin derives from `DAQ_Viewer_base` and defines `ini_attributes`, `ini_detector`, `commit_settings` | the mixin calls `super()` on them | all PyMoDAQ viewer plugins have them |
| 2 | New frames go through a method `emit_data(frame)` where `frame` is an array whose **last two dimensions are (y, x)** (e.g. `(h, w)` or `(n_average, n_frames, h, w)`; leading dimensions are averaged) | `emit_data` | see step 3 (the frame is emitted elsewhere) |
| 3 | A parameter `timing_opts` / `exposure_time` in **ms** | timestamp of the delay (middle of the exposure) | if missing, the exposure is taken as 0 |
| 4 | *(optional)* attributes `x_axis` and `y_axis` (`Axis` of the image, with `offset` and `scaling`) | converts the ROI drawn in the image viewer to pixels | if missing, the ROI is taken in pixels |
| 5 | *(optional)* a method `update_fps()` and a parameter `timing_opts` / `fps_on` | FPS display | only used if both exist |
| 6 | No existing parameter group named `fringe` | the mixin adds a group with that name | rename your group |

Grayscale frames only. For colour cameras convert the frame to a 2D array before calling the mixin (see step 3).

## 2. Create the plugin file

In your plugin package, create `daq_viewer_plugins/plugins_0D/daq_0Dviewer_<Name>.py`. The **file name**
`daq_0Dviewer_<Name>` and the **class name** `DAQ_0DViewer_<Name>` must match (PyMoDAQ rule), and the plugin must be
in `plugins_0D` so that it is listed as a 0D detector in the dashboard.

```python
from pymodaq.control_modules.viewer_utility_classes import main

from pymodaq_plugins_ramping_xuv_sim.hardware.fringe_delay import FringeDelayMixin
# the existing camera plugin (any package, it only has to be installed)
from pymodaq_plugins_mycamera.daq_viewer_plugins.plugins_2D.daq_2Dviewer_MyCamera import DAQ_2DViewer_MyCamera


class DAQ_0DViewer_MyCamera_FringeDelay(FringeDelayMixin, DAQ_2DViewer_MyCamera):
    """ Delay reader from the fringes seen by MyCamera. See FringeDelayMixin."""
    params = DAQ_2DViewer_MyCamera.params + FringeDelayMixin.fringe_params


if __name__ == '__main__':
    main(__file__)
```

Two rules:

* The **mixin comes first** in the list of base classes (so that its `emit_data`, `commit_settings`, `roi_select`
  and `ini_detector` are the ones called, and they call the camera's through `super()`).
* The **parameters are concatenated**: `camera.params + FringeDelayMixin.fringe_params`.

If the camera plugin needs `main(__file__, init=False)` (as the Thorlabs one does), keep it.

Declare the camera package as a dependency of your plugin package (`pyproject.toml`, `dependencies`) and
reinstall (`pip install -e .`) so that PyMoDAQ finds the new plugin.

## 3. If the camera plugin does not call `emit_data(frame)`

Some plugins emit the image directly in `grab_data` or in a callback. Then do not rely on the automatic
interception: find the place where the plugin emits the image, and replace

```python
self.dte_signal.emit(DataToExport('Camera', data=[DataFromPlugins(name='Camera', data=[image], dim='Data2D', ...)]))
```

by (in your subclass, or by editing the emission point to call a method of your subclass)

```python
import time
self.dte_signal.emit(self.analyse_image(image, time.time()))   # image: 2D numpy array
```

`analyse_image(image, t_received)` is the method used by the mixin: it returns the complete `DataToExport`
(delay, visibility and diagnostics). A minimal way to do it without touching the camera plugin:

```python
class DAQ_0DViewer_MyCamera_FringeDelay(FringeDelayMixin, DAQ_2DViewer_MyCamera):
    params = DAQ_2DViewer_MyCamera.params + FringeDelayMixin.fringe_params

    def my_frame_ready_callback(self, frame):          # the name of the hook of *your* camera plugin
        image = np.asarray(frame, dtype=float)
        if image.ndim == 3:                             # colour: convert to grayscale
            image = image.mean(axis=-1)
        self.dte_signal.emit(self.analyse_image(image, time.time()))
```

## 4. Use it in the dashboard

1. Add the module in your preset (or in the dashboard): detector type **DAQ0D**, plugin
   `MyCamera_FringeDelay`. Configure the camera as you always do (serial number, exposure, gain...) and initialise.
2. In *Fringe analysis*, set the **Wavelength (nm)** (it defines the delay of one optical cycle, 2.64 fs at
   790 nm) and the **Sign** (+/-) of the delay with respect to the phase.
3. Start a **continuous grab**. Three extra viewers appear next to the delay trace: *Camera image*,
   *Fourier transform* (and *Fringes* if you enable it in *Display*).
4. **Select the fringes**: in the *Camera image* viewer click the ROI tool (dashed rectangle) and place the
   rectangle on a region with clean fringes. With *Use viewer ROI* checked, `x0, y0, width, height` are
   updated. Set **Fringes along** to `x` if the intensity varies along x (vertical fringes), `y` otherwise.
   Width/height 0 means the whole image.
5. **Select the peak**: by default the strongest peak of the Fourier transform is used (*Automatic*), its
   indices are shown in *First index / Last index*. To force it, use the ROI tool (linear region) of the
   *Fourier transform* viewer around the fringe peak: *Automatic* is then unchecked. The
   *Fringes* trace should look like a clean cosine.
6. Check the **visibility** (0 to 1) and the **Tracking** LED. Adjust the **Visibility threshold**: below it the
   delay is NaN (the phase is not updated, the extension ignores NaN delays).
7. Press **Set delay to zero** when the interferometer is at the delay you want to be 0 (for instance time
   zero found with the signal). **Reset tracking** restarts the unwrapping (use it after losing the signal
   for a while or after moving the stage by a lot).
8. Hide the diagnostics you don't need in *Display* (the image is the heaviest one).

Use the module as delay detector of the **Delay Binner** extension (its name should contain "delay" to be proposed
automatically).

## 5. Settings that matter

* **Speed limit**: the phase is unwrapped with respect to the previous frame, so the delay must change by less than
  half a cycle between two frames: `speed < wavelength / (2 c) * frame rate`. At 790 nm and 100 frames/s this is
  130 fs/s. A slow piezo is far below, but check with the camera rate if you use the frame rate limited by
  your exposure.
* **Exposure**: no saturation and a fringe contrast kept (a fast motion during the exposure washes out the fringes:
  the visibility drops).
* **Timestamp latency**: delays are timestamped with the **middle of the exposure** (`reception time - latency -
  exposure/2`). Set *Timestamp latency (ms)* to the time between the end of the exposure and the reception of the
  frame (readout + transfer + callback) if you know it: it improves the matching with the spectra in the
  Delay Binner.
* **Hardware ROI** of the camera (if the plugin has one) reduces the frame size and speeds up the analysis.
* The delay is known **modulo one optical cycle at the start** of the tracking (the first phase is taken in
  (-pi, pi]); only *changes* are tracked beyond a cycle. The absolute zero comes from *Set delay to zero*.

## 6. Test it without the real hardware, then with it

1. Run the unit tests of the analysis: `pytest tests/test_fringe_phase.py`.
2. Compare with the simulated version: the dashboard module `FringeDelaySim` uses the same mixin with a simulated
   camera (`DAQ_2DViewer_FringeCameraSim`) whose fringes follow the simulated piezo. Reproducing the behaviour
   of your plugin with it (same parameters) is a good check.
3. With the real camera, validate in this order: *image viewer* shows fringes, *fringes* trace is a clean
   cosine, *Fourier transform* shows one peak, visibility is above the threshold, and the delay changes
   smoothly (no steps of one cycle = `wavelength / c`) when you move the piezo slowly in both directions. A
   delay that jumps by exactly one cycle means the speed limit or the noise is too high (lower the speed, increase
   the exposure or the ROI).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| The delay stays at 0 / the module shows the camera image only | the plugin does not call `emit_data(frame)`: see step 3 |
| `AttributeError`/`KeyError` in the status bar at every frame | `timing_opts` / `fps_on` missing while `update_fps` exists: add the group or remove `update_fps` |
| The plugin is not in the DAQ0D list | file/class name mismatch, file not in `plugins_0D`, camera package not installed, or package not reinstalled |
| Delay always NaN | visibility below the threshold: wrong ROI (outside the fringes, or *Fringes along* wrong), saturation, or low contrast |
| Delay jumps by one cycle | speed or noise: see the speed limit; check the FFT peak is not switching between two peaks (set it manually) |
| Delay noisy | small ROI: enlarge it; narrower peak band (*Half width* 0 or 1); longer exposure |
| ROI drawn in the viewer has no effect | *Use viewer ROI* unchecked, or the ROI drawn in a viewer other than *Camera image* / *Fourier transform* |
