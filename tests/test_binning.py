import numpy as np
import pytest

from pymodaq_plugins_ramping_xuv_sim.binning import DelayBinner

ENERGY = np.linspace(30, 60, 4)


def spec(value):
    return np.full(4, float(value))


def test_basic_binning_and_mean():
    b = DelayBinner(0., 4., 4, window=0.1)
    for t in np.arange(0, 1, 0.01):  # delay measurements ramp 0 -> 4 in 1 s
        b.add_delay(t, 4 * t)
    b.add_spectrum(0.125, spec(1.), ENERGY)  # delay 0.5 -> bin 0
    b.add_spectrum(0.15, spec(3.), ENERGY)  # delay 0.6 -> bin 0
    b.add_spectrum(0.625, spec(10.), ENERGY)  # delay 2.5 -> bin 2
    b.add_delay(1.0, 4.)  # makes sure the last window is complete
    assert list(b.counts) == [2, 0, 1, 0]
    m = b.map
    assert m.shape == (4, 4)
    assert np.allclose(m[0], 2.) and np.allclose(m[2], 10.)
    assert np.all(np.isnan(m[1])) and np.all(np.isnan(m[3]))
    assert np.allclose(b.centers, [0.5, 1.5, 2.5, 3.5])


def test_spectrum_waits_for_late_delay_measurements():
    b = DelayBinner(0., 4., 4, window=0.2)
    b.add_delay(0.0, 1.)
    b.add_spectrum(1.0, spec(5.))  # window [0.9, 1.1]: no delay data yet
    assert b.n_used == 0
    b.add_delay(0.95, 3.2)
    assert b.n_used == 0  # window not complete
    b.add_delay(1.2, 3.2)
    assert b.n_used == 1 and b.counts[3] == 1


def test_out_of_range_and_no_delay():
    b = DelayBinner(0., 4., 4, window=0.1)
    b.add_delay(0., 10.)
    b.add_delay(0.1, 10.)
    b.add_spectrum(0.05, spec(1.))
    assert b.n_out_of_range == 1 and b.n_used == 0
    b.add_spectrum(500., spec(1.))
    b.add_delay(501., 1.)
    assert b.n_no_delay == 1  # no delay measurement anywhere near t=500 s... 
    assert b.n_used == 0


def test_reset_changes_bins():
    b = DelayBinner(0., 4., 4, window=0.1)
    b.add_delay(0., 1.)
    b.add_spectrum(0., spec(1.))
    b.add_delay(1., 1.)
    assert b.n_used == 1
    b.reset(-1., 1., 10)
    assert b.n_bins == 10 and b.n_used == 0 and b.map is None


def test_old_delay_measurements_are_dropped():
    b = DelayBinner(0., 4., 4, window=0.1)
    for t in np.arange(0, 100, 0.01):
        b.add_delay(t, 1.)
        if int(t * 100) % 100 == 0:
            b.add_spectrum(t, spec(1.))
    assert len(b._t_delay) < 500
    assert b.n_used > 90


def test_to_uniform_axis():
    from pymodaq_plugins_ramping_xuv_sim.binning import to_uniform_axis
    energy = np.array([60., 50., 41., 40., 30.])  # decreasing, non uniform
    map_ = np.array([energy * 2, energy * -1])
    new_map, new_energy = to_uniform_axis(map_, energy, 50)
    assert new_energy[0] == 30 and new_energy[-1] == 60
    assert np.allclose(np.diff(new_energy), np.diff(new_energy)[0])
    assert new_map.shape == (2, 50)
    assert np.interp(40., new_energy, new_map[0]) == pytest.approx(80., abs=1e-6)
    assert np.interp(50., new_energy, new_map[1]) == pytest.approx(-50., abs=1.0)
