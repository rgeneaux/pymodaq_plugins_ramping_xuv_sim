import numpy as np
import pytest

from pymodaq_plugins_ramping_xuv_sim.hardware.fringe_phase import FringePhaseTracker, C

WAVELENGTH = 790.
CYCLE_FS = WAVELENGTH * 1e-9 / C * 1e15  # 2.635 fs


def fringes(delay_fs, n=128, period=12.3, visibility=0.6, noise=0., rng=None):
    phase = 2 * np.pi * delay_fs / CYCLE_FS
    x = np.arange(n)
    out = 1000 * (1 + visibility * np.cos(2 * np.pi * x / period + phase))
    if noise:
        out = out + (rng or np.random.default_rng(0)).normal(0, noise, n)
    return out


def test_tracks_delay_over_several_cycles():
    tracker = FringePhaseTracker(WAVELENGTH)
    delays = np.linspace(-7, 7, 700)  # 5 optical cycles, 0.02 fs steps
    out = np.array([tracker.process(fringes(d)).delay for d in delays])
    err = out - delays
    assert np.ptp(err) < 0.02  # constant offset (unknown starting phase), no cycle slip
    # going back down gives the same delays
    back = np.array([tracker.process(fringes(d)).delay for d in delays[::-1]])
    assert np.allclose(back - delays[::-1], err[-1], atol=0.02)


def test_no_unwrapping_without_tracking_would_fail():
    """Sanity check of the test: the wrapped phase alone is bounded by half a cycle"""
    tracker = FringePhaseTracker(WAVELENGTH)
    wrapped = [np.angle(np.exp(1j * 2 * np.pi * d / CYCLE_FS)) for d in np.linspace(-7, 7, 100)]
    assert np.ptp(wrapped) <= 2 * np.pi + 1e-9


def test_set_zero_and_reset():
    tracker = FringePhaseTracker(WAVELENGTH)
    for d in np.linspace(0, 3, 150):
        tracker.process(fringes(d))
    tracker.set_zero()
    r = tracker.process(fringes(3.0))
    assert r.delay == pytest.approx(0., abs=0.02)
    r = tracker.process(fringes(3.5))
    assert r.delay == pytest.approx(0.5, abs=0.02)
    tracker.reset()
    first = tracker.process(fringes(3.5))
    assert -np.pi < first.phase <= np.pi  # new tracking starts in (-pi, pi]
    assert tracker.process(fringes(4.0)).delay - first.delay == pytest.approx(0.5, abs=0.02)


def test_sign():
    a, b = FringePhaseTracker(WAVELENGTH), FringePhaseTracker(WAVELENGTH, sign=-1)
    for d in np.linspace(0, 1, 50):
        ra, rb = a.process(fringes(d)), b.process(fringes(d))
    assert ra.delay == pytest.approx(-rb.delay)


def test_noise():
    rng = np.random.default_rng(1)
    tracker = FringePhaseTracker(WAVELENGTH)
    delays = np.linspace(0, 6, 400)
    out = np.array([tracker.process(fringes(d, noise=3, rng=rng)).delay for d in delays])
    assert np.std(out - delays) < 0.01


def test_weak_fringes_are_rejected_and_tracking_resumes():
    tracker = FringePhaseTracker(WAVELENGTH, visibility_threshold=0.1)
    for d in np.linspace(0, 2, 100):
        good = tracker.process(fringes(d))
    offset = good.delay - 2.
    bad = tracker.process(fringes(2.05, visibility=0.02))
    assert not bad.valid and np.isnan(bad.delay) and bad.visibility < 0.1
    resumed = tracker.process(fringes(2.1))
    assert resumed.valid and resumed.delay - 2.1 == pytest.approx(offset, abs=0.02)


def test_visibility_value_and_auto_band():
    tracker = FringePhaseTracker(WAVELENGTH)
    res = tracker.process(fringes(0., visibility=0.6, period=128 / 10))  # exactly 10 periods in 128 px
    assert res.visibility == pytest.approx(0.6, abs=0.02)
    assert res.band[0] <= 10 <= res.band[1]
    manual = FringePhaseTracker(WAVELENGTH, band=(12, 8))
    assert manual.process(fringes(0.)).band == (8, 12)


def test_fringes_from_image_roi_and_direction():
    x = np.arange(100)
    image = np.tile(1 + np.cos(2 * np.pi * x / 10), (60, 1))  # fringes along x
    image[:20] = 0  # something else outside the roi
    f = FringePhaseTracker.fringes_from_image(image, roi=(10, 20, 50, 0))
    assert f.shape == (50,) and np.allclose(f, image[30, 10:60])
    g = FringePhaseTracker.fringes_from_image(image.T, roi=None, along='y')
    assert g.shape == (100,)
