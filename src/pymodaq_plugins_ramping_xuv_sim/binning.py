"""Core (Qt free) of the delay binning: matches asynchronous dR/R spectra with asynchronous delay measurements and
accumulates the spectra in a 2D (delay bin x photon energy) map.
"""
from typing import Optional

import numpy as np


class DelayBinner:
    """Average spectra into pre-defined delay bins

    Spectra and delay measurements arrive independently (each with the timestamp of the measurement; for a
    spectrum, the middle of its integration). The delay attributed to a spectrum is the mean of the measured delays
    within ``window`` seconds centered on the spectrum timestamp. The spectrum is then added to the bin containing
    this delay (bins are uniform between ``delay_min`` and ``delay_max``). The 2D map is the mean spectrum of each bin.

    Parameters
    ----------
    delay_min, delay_max: float
        Edges (same unit as the measured delays, fs) of the first and last bins
    n_bins: int
    window: float
        Duration (s) of the integration of a spectrum, over which the delay measurements are averaged

    Attributes
    ----------
    n_used, n_out_of_range, n_no_delay: int
        Number of spectra added to the map, outside the delay range, and without any delay measurement
        in their window
    """
    def __init__(self, delay_min: float, delay_max: float, n_bins: int, window: float):
        self.window = window
        self._set_bins(delay_min, delay_max, n_bins)
        self._t_delay: list = []
        self._delay: list = []
        self._pending: list = []  # (timestamp, spectrum) waiting for the delay measurements covering their window
        self.energy: Optional[np.ndarray] = None
        self.reset()

    def _set_bins(self, delay_min: float, delay_max: float, n_bins: int):
        if n_bins < 1 or delay_max <= delay_min:
            raise ValueError('Need at least one bin and delay_max > delay_min')
        self.edges = np.linspace(delay_min, delay_max, int(n_bins) + 1)

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    def reset(self, delay_min: float = None, delay_max: float = None, n_bins: int = None):
        """Empty the map (and eventually change the bins)"""
        if n_bins is not None:
            self._set_bins(self.edges[0] if delay_min is None else delay_min,
                           self.edges[-1] if delay_max is None else delay_max, n_bins)
        self._sum: Optional[np.ndarray] = None
        self.counts = np.zeros(self.n_bins, dtype=int)
        self._pending = []
        self.n_used = self.n_out_of_range = self.n_no_delay = 0

    def add_delay(self, timestamp: float, delay: float):
        """Add a delay measurement"""
        self._t_delay.append(timestamp)
        self._delay.append(delay)
        self._process()

    def add_spectrum(self, timestamp: float, spectrum: np.ndarray, energy: np.ndarray = None):
        """Add a spectrum, measured at timestamp (middle of its integration)"""
        if energy is not None:
            self.energy = np.asarray(energy)
        spectrum = np.asarray(spectrum, dtype=float)
        if self._sum is None or self._sum.shape[1] != spectrum.size:
            self._sum = np.zeros((self.n_bins, spectrum.size))
        self._pending.append((timestamp, spectrum))
        self._process()

    def _process(self):
        """Bin the pending spectra for which all the delay measurements of the window have arrived"""
        if not self._pending or not self._t_delay:
            return
        t_delay, delay = np.asarray(self._t_delay), np.asarray(self._delay)
        half = self.window / 2
        still_pending = []
        for t, spectrum in self._pending:
            if t + half > t_delay[-1]:
                still_pending.append((t, spectrum))
                continue
            i0, i1 = np.searchsorted(t_delay, [t - half, t + half], side='right')
            if i1 <= i0:  # no measurement in the window: take the closest one if reasonably close
                i = int(np.argmin(np.abs(t_delay - t)))
                if abs(t_delay[i] - t) > max(self.window, 0.1):
                    self.n_no_delay += 1
                    continue
                i0, i1 = i, i + 1
            self._bin(float(delay[i0:i1].mean()), spectrum)
        self._pending = still_pending

        # forget the delay measurements that cannot be needed anymore
        t_keep = min([t for t, _ in self._pending], default=t_delay[-1]) - half - 1.
        n_old = int(np.searchsorted(t_delay, t_keep))
        if n_old > 0:
            del self._t_delay[:n_old]
            del self._delay[:n_old]

    def _bin(self, delay: float, spectrum: np.ndarray):
        if not (self.edges[0] <= delay < self.edges[-1]):
            self.n_out_of_range += 1
            return
        ind = int(np.searchsorted(self.edges, delay, side='right')) - 1
        self._sum[ind] += spectrum
        self.counts[ind] += 1
        self.n_used += 1

    @property
    def map(self) -> Optional[np.ndarray]:
        """Mean spectrum of each bin, shape (n_bins, n_energy); NaN for the empty bins. None before any spectrum."""
        if self._sum is None:
            return None
        with np.errstate(invalid='ignore', divide='ignore'):
            return self._sum / self.counts[:, None]


def to_uniform_axis(map_: np.ndarray, energy: np.ndarray, n_points: int = None):
    """Resample the last axis of ``map_`` on a uniform grid (the 2D viewers need linear axes)

    Parameters
    ----------
    map_: np.ndarray
        shape (n_bins, len(energy))
    energy: np.ndarray
        monotonic (increasing or decreasing), possibly non uniform axis
    n_points: int
        number of points of the uniform axis, default twice the number of points of energy

    Returns
    -------
    map, energy: uniformly sampled, with an increasing energy axis
    """
    energy = np.asarray(energy, dtype=float)
    if energy[0] > energy[-1]:
        energy, map_ = energy[::-1], map_[:, ::-1]
    n_points = n_points or 2 * len(energy)
    new_energy = np.linspace(energy[0], energy[-1], n_points)
    return np.array([np.interp(new_energy, energy, row) for row in map_]), new_energy
