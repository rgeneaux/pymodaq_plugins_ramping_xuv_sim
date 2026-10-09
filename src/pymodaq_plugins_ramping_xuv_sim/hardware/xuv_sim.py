"""Simulated attosecond transient-reflectivity experiment.

One object plays the role of the whole setup: a piezo delay stage, an optical delay measurement (interferometer) and
an XUV spectrometer that returns dR/R as a function of photon energy at the current delay. The dR/R map is an
experimental trace stored in ``resources/MgO_Yb_proc.npz``.

The stage can either go to a position (finite velocity) or oscillate between two positions (triangular waveform).
The delay really seen by the sample ("true delay") is not exactly the position the piezo claims: it contains a
periodic non-linearity and a drift. In the real experiment it is measured optically (fringe phase), which is simulated
by :meth:`XUVExperimentSim.measure_delay` (an interferometer with its own noise).

The stage, the spectrometer and the interferometer are different PyMoDAQ plugins (running in different threads of
the same process), so they all use the same instance obtained from :meth:`XUVExperimentSim.shared`.
All times are ``time.time()`` (same clock as the PyMoDAQ data timestamps).
"""
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

DATA_PATH = Path(__file__).parent.parent.joinpath('resources', 'MgO_Yb_proc.npz')


class XUVExperimentSim:
    """Piezo delay stage + interferometer + dR/R spectrometer, built from an experimental (delay, energy) map.

    Parameters
    ----------
    data_path: Path
        npz file with ``delay`` (fs, N), ``E`` (eV, M) and ``dR`` (N x M)

    Attributes
    ----------
    velocity: float
        Piezo velocity in fs/s for point to point moves
    ripple_amplitude, ripple_period: float
        Periodic non-linearity (fs) of the piezo: true delay = position + amplitude * sin(2 pi position / period)
    drift_rate: float
        Linear drift (fs/s) of the true delay
    """
    units = 'fs'
    _shared: Optional['XUVExperimentSim'] = None
    _shared_lock = threading.Lock()

    def __init__(self, data_path: Path = DATA_PATH):
        with np.load(data_path) as data:
            delay, energy, dr = data['delay'], data['E'], data['dR']
        # sort both axes ascending (the file stores them descending)
        i_delay, i_energy = np.argsort(delay), np.argsort(energy)
        self.delay = delay[i_delay]
        self.energy = energy[i_energy]
        self.dR = dr[i_delay][:, i_energy]

        self.velocity = 20.  # fs/s
        self.min_position = float(self.delay[0])
        self.max_position = float(self.delay[-1])

        self.ripple_amplitude = 0.1
        self.ripple_period = 1.7
        self.drift_rate = 0.

        self._lock = threading.RLock()
        self._origin = 0.  # position when the last move started
        self._target = 0.
        self._t_start = time.time()
        self._t_created = self._t_start

        self._oscillating = False
        self._osc_min, self._osc_max, self._osc_velocity, self._osc_t0 = 0., 0., 1., 0.

    @classmethod
    def shared(cls) -> 'XUVExperimentSim':
        """The instance shared by the stage, the spectrometer and the interferometer plugins"""
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    # ---- piezo stage -------------------------------------------------------------------------------------------
    def _position_at(self, t: float) -> float:
        """Position claimed by the piezo at time t (waveform or constant velocity move)"""
        if self._oscillating:
            length = self._osc_max - self._osc_min
            s = (self._osc_velocity * (t - self._osc_t0)) % (2 * length)
            return self._osc_min + (s if s <= length else 2 * length - s)
        distance = self._target - self._origin
        travelled = self.velocity * max(t - self._t_start, 0.)
        if travelled >= abs(distance):
            return self._target
        return self._origin + np.sign(distance) * travelled

    def true_delay_at(self, t: float) -> float:
        """Delay really seen by the sample at time t: piezo position + non-linearity + drift"""
        x = self._position_at(t)
        return float(x + self.ripple_amplitude * np.sin(2 * np.pi * x / self.ripple_period)
                     + self.drift_rate * (t - self._t_created))

    def get_position(self) -> float:
        with self._lock:
            return float(self._position_at(time.time()))

    def move_at(self, target: float):
        """Start moving to target (fs), clipped to the stage range. Returns immediately (stops any oscillation)."""
        with self._lock:
            now = time.time()
            self._origin = self._position_at(now)
            self._oscillating = False
            self._target = float(np.clip(target, self.min_position, self.max_position))
            self._t_start = now

    def start_oscillation(self, minimum: float, maximum: float, velocity: float):
        """Oscillate with a triangular waveform between minimum and maximum (fs), starting at minimum going up"""
        with self._lock:
            lo, hi = sorted((minimum, maximum))
            self._osc_min = float(np.clip(lo, self.min_position, self.max_position))
            self._osc_max = float(np.clip(hi, self.min_position, self.max_position))
            self._osc_velocity = float(abs(velocity))
            self._osc_t0 = time.time()
            self._oscillating = self._osc_max > self._osc_min and self._osc_velocity > 0

    @property
    def oscillating(self) -> bool:
        return self._oscillating

    def stop(self):
        """Stop the piezo where it is (also stops an oscillation)"""
        with self._lock:
            now = time.time()
            self._origin = self._target = self._position_at(now)
            self._oscillating = False
            self._t_start = now

    def is_moving(self) -> bool:
        with self._lock:
            return self._oscillating or self._position_at(time.time()) != self._target

    # ---- interferometer ----------------------------------------------------------------------------------------
    def measure_delay(self, noise: float = 0., rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
        """Optical measurement (fringe phase) of the true delay, now.

        Returns
        -------
        timestamp: float
            time.time() of the measurement
        delay: float
            measured delay in fs (true delay + gaussian noise of rms ``noise``)
        """
        rng = rng or np.random.default_rng()
        with self._lock:
            t = time.time()
            delay = self.true_delay_at(t)
        return t, delay + float(rng.normal(0., noise))

    # ---- spectrometer ------------------------------------------------------------------------------------------
    def spectrum_at(self, delays) -> np.ndarray:
        """Noise-free dR/R spectrum at the given delay(s), linearly interpolated, clipped to the measured range.

        Returns an array of shape (len(delays), M) for array input, (M,) for a scalar.
        """
        scalar = np.ndim(delays) == 0
        d = np.clip(np.atleast_1d(delays).astype(float), self.delay[0], self.delay[-1])
        i = np.clip(np.searchsorted(self.delay, d) - 1, 0, len(self.delay) - 2)
        w = ((d - self.delay[i]) / (self.delay[i + 1] - self.delay[i]))[:, None]
        out = (1 - w) * self.dR[i] + w * self.dR[i + 1]
        return out[0] if scalar else out

    def measure(self, exposure: float, noise_1s: float, n_average: int = 1,
                n_sub: int = 5, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, float]:
        """Simulate an integration of ``exposure`` seconds starting now (the call does not sleep).

        The signal is averaged over the true delays visited by the stage during the exposure (so a fast sweep blurs
        the trace) and Gaussian noise is added whose rms is ``noise_1s / sqrt(exposure * n_average)``.

        Returns
        -------
        spectrum: np.ndarray
            dR/R, same length as :attr:`energy`
        delay: float
            mean true delay (fs) of the exposure
        """
        rng = rng or np.random.default_rng()
        t0 = time.time()
        with self._lock:
            delays = np.array([self.true_delay_at(t0 + f * exposure) for f in np.linspace(0, 1, n_sub)])
        spectrum = self.spectrum_at(delays).mean(axis=0)
        sigma = noise_1s / np.sqrt(max(exposure, 1e-6) * max(n_average, 1))
        return spectrum + rng.normal(0., sigma, spectrum.shape), float(delays.mean())
