"""Phase of a fringe pattern, tracked continuously to get the delay of an interferometer beyond one optical cycle.

The logic follows the PID model of the spatial interferometry stabilisation (pymodaq_plugins_PIDStabFringes): the
image is averaged over the rows of a region of interest to get 1D fringes, the phase is the one of the peak of
their Fourier transform, and it is unwrapped with respect to the previously tracked phase so that the 2 pi jumps
are accounted for. The delay is then ``phase * wavelength / (2 pi c)``.

This works as long as the delay changes by less than half a cycle (wavelength / (2 c), 1.3 fs at 790 nm) between two
frames.
"""
import dataclasses
from typing import Optional, Tuple

import numpy as np

C = 299792458.  # m/s


@dataclasses.dataclass
class FringeResult:
    """Result of the analysis of one image"""
    delay: float  # fs, NaN if the fringes are not visible enough
    phase: float  # rad, unwrapped (continuous) phase, relative to the zero
    visibility: float  # fringe contrast of the peak, between 0 and 1
    valid: bool  # True if the visibility is above the threshold (the phase has then been tracked)
    fringes: np.ndarray  # 1D fringes (mean of the ROI)
    spectrum: np.ndarray  # modulus of the Fourier transform of the fringes
    band: Tuple[int, int]  # first and last (included) index of the Fourier transform used for the phase


class FringePhaseTracker:
    """Extract and track the phase of fringes

    Parameters
    ----------
    wavelength: float
        Wavelength in nm, giving the delay corresponding to a phase
    band: tuple of int or None
        First and last (included) indices of the Fourier transform peak. None for an automatic detection of the
        strongest peak (excluding the lowest ``min_index`` indices) +/- ``half_width``
    visibility_threshold: float
        Below this fringe visibility the frame is rejected (the phase is not updated)
    sign: int
        +1 or -1 to choose the sign of the delay with respect to the phase
    """
    def __init__(self, wavelength: float = 790., band: Optional[Tuple[int, int]] = None, half_width: int = 1,
                 visibility_threshold: float = 0.05, sign: int = 1, min_index: int = 3):
        self.wavelength = wavelength
        self.band = band
        self.half_width = half_width
        self.visibility_threshold = visibility_threshold
        self.sign = sign
        self.min_index = min_index
        self.reset()

    def reset(self):
        """Forget the tracked phase: the next valid frame starts a new tracking (its phase in (-pi, pi])"""
        self._phase: Optional[float] = None  # unwrapped phase of the last valid frame
        self._offset = 0.  # phase of the delay zero

    def set_zero(self):
        """The current delay becomes the zero delay"""
        if self._phase is not None:
            self._offset = self._phase

    @property
    def phase_to_delay(self) -> float:
        """fs per rad"""
        return self.sign * self.wavelength * 1e-9 / (2 * np.pi * C) * 1e15

    @staticmethod
    def fringes_from_image(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None,
                           along: str = 'x') -> np.ndarray:
        """1D fringes: mean of the image in the ROI (x0, y0, width, height), over the direction perpendicular to
        the direction ``along`` of the fringe modulation. A width or height of 0 means the whole image."""
        image = np.asarray(image, dtype=float)
        if roi is not None:
            x0, y0, w, h = roi
            x1 = x0 + w if w > 0 else image.shape[1]
            y1 = y0 + h if h > 0 else image.shape[0]
            image = image[max(y0, 0):y1, max(x0, 0):x1]
        return np.nanmean(image, axis=0 if along == 'x' else 1)

    def process(self, fringes: np.ndarray) -> FringeResult:
        """Phase and delay from 1D fringes"""
        fringes = np.asarray(fringes, dtype=float)
        mean = np.nanmean(fringes)
        window = np.hanning(len(fringes))  # limits the leakage of the negative frequency peak on the positive one
        tf = np.fft.rfft((fringes - mean) * window)
        modulus = np.abs(tf)
        band = self._get_band(modulus)
        peak = tf[band[0]:band[1] + 1]

        # amplitude of the cosine at the peak relative to the mean intensity (1 for perfect fringes)
        visibility = 2 * np.max(np.abs(peak)) / (np.sum(window) * mean) if mean > 0 and len(peak) > 0 else 0.
        valid = bool(len(peak) > 0 and visibility >= self.visibility_threshold)
        if not valid:
            return FringeResult(np.nan, np.nan if self._phase is None else self._phase - self._offset,
                                visibility, False, fringes, modulus, band)

        wrapped = float(np.angle(np.sum(peak)))  # amplitude weighted, no problem when angles straddle +/- pi
        if self._phase is None:
            self._phase = wrapped
        else:  # continuity: the new phase is the closest one to the previous one (modulo 2 pi)
            self._phase += float(np.angle(np.exp(1j * (wrapped - self._phase))))
        phase = self._phase - self._offset
        return FringeResult(phase * self.phase_to_delay, phase, visibility, True, fringes, modulus, band)

    def _get_band(self, modulus: np.ndarray) -> Tuple[int, int]:
        if self.band is not None:
            first, last = sorted(int(b) for b in self.band)
            return max(first, 0), min(last, len(modulus) - 1)
        if len(modulus) <= self.min_index + 1:
            return 0, 0
        k = self.min_index + int(np.argmax(modulus[self.min_index:]))
        return max(k - self.half_width, 1), min(k + self.half_width, len(modulus) - 1)
