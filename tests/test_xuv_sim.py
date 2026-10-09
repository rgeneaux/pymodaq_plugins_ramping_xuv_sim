import time

import numpy as np
import pytest

from pymodaq_plugins_ramping_xuv_sim.hardware.xuv_sim import XUVExperimentSim


@pytest.fixture
def sim():
    s = XUVExperimentSim()
    s.ripple_amplitude = 0.  # true delay == piezo position
    return s


def test_axes_sorted(sim):
    assert np.all(np.diff(sim.delay) > 0) and np.all(np.diff(sim.energy) > 0)
    assert sim.dR.shape == (len(sim.delay), len(sim.energy))


def test_spectrum_on_grid_and_midpoint(sim):
    assert np.allclose(sim.spectrum_at(sim.delay[10]), sim.dR[10])
    mid = 0.5 * (sim.delay[10] + sim.delay[11])
    assert np.allclose(sim.spectrum_at(mid), 0.5 * (sim.dR[10] + sim.dR[11]))


def test_spectrum_clipped_outside_range(sim):
    assert np.allclose(sim.spectrum_at(-100.), sim.dR[0])
    assert np.allclose(sim.spectrum_at(100.), sim.dR[-1])


def test_move_finite_velocity(sim):
    sim.velocity = 50.
    sim.move_at(2.)
    time.sleep(0.02)
    assert 0. < sim.get_position() < 2.
    assert sim.is_moving()
    time.sleep(0.1)
    assert sim.get_position() == pytest.approx(2.)
    assert not sim.is_moving()
    sim.move_at(1000.)  # clipped to stage range
    sim.velocity = 1e6
    time.sleep(0.01)
    assert sim.get_position() == pytest.approx(sim.max_position)


def test_stop(sim):
    sim.velocity = 10.
    sim.move_at(5.)
    time.sleep(0.05)
    sim.stop()
    pos = sim.get_position()
    time.sleep(0.05)
    assert sim.get_position() == pos


def test_noise_decreases_with_exposure(sim):
    rng = np.random.default_rng(0)
    sim.move_at(0.)
    noise = {}
    for exposure in (0.01, 1.):
        spectra = np.array([sim.measure(exposure, 1e-2, rng=rng)[0] for _ in range(300)])
        noise[exposure] = spectra.std(axis=0).mean()
    assert noise[0.01] / noise[1.] == pytest.approx(10., rel=0.15)
    assert noise[1.] == pytest.approx(1e-2, rel=0.15)


def test_measure_follows_stage(sim):
    sim.velocity = 1e6
    sim.move_at(-3.)
    time.sleep(0.01)
    spectrum, delay = sim.measure(0.1, 0.)
    assert delay == pytest.approx(-3.)
    assert np.allclose(spectrum, sim.spectrum_at(-3.))


def test_oscillation_triangular(sim):
    sim.start_oscillation(-2., 2., 100.)  # period 0.08 s
    assert sim.oscillating and sim.is_moving()
    t0 = sim._osc_t0
    assert sim._position_at(t0) == pytest.approx(-2., abs=1e-3)
    assert sim._position_at(t0 + 0.02) == pytest.approx(0., abs=1e-3)
    assert sim._position_at(t0 + 0.04) == pytest.approx(2., abs=1e-3)
    assert sim._position_at(t0 + 0.06) == pytest.approx(0., abs=1e-3)
    assert sim._position_at(t0 + 0.08) == pytest.approx(-2., abs=1e-3)
    sim.stop()
    assert not sim.oscillating
    sim.start_oscillation(-2., 2., 1.)
    sim.move_at(1.)  # a move cancels the waveform
    assert not sim.oscillating


def test_true_delay_differs_from_piezo_position(sim):
    sim.ripple_amplitude = 0.1
    sim.move_at(0.4)
    sim.velocity = 1e6
    time.sleep(0.01)
    t, delay = sim.measure_delay(noise=0.)
    assert delay != pytest.approx(sim.get_position(), abs=1e-3)
    assert delay == pytest.approx(sim.true_delay_at(t), abs=1e-6)
    assert abs(delay - sim.get_position()) <= 0.1 + 1e-9
