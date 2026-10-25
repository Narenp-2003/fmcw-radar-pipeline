import numpy as np
from scipy.signal.windows import hann

from radar.params import RadarParams


def range_doppler_map(frame, window=True):
    """Complex range-Doppler map, shape (n_chirps, n_samples).

    Axis 0 is Doppler (fftshifted, zero velocity at n_chirps // 2).
    Axis 1 is range (bin i is at i * range_resolution).
    """
    m, n = frame.shape
    x = frame
    if window:
        x = x * hann(n, sym=False)[None, :] * hann(m, sym=False)[:, None]
    rd = np.fft.fft(x, axis=1)
    rd = np.fft.fft(rd, axis=0)
    return np.fft.fftshift(rd, axes=0)


def range_axis(p: RadarParams):
    return np.arange(p.n_samples) * p.range_resolution


def velocity_axis(p: RadarParams):
    return (np.arange(p.n_chirps) - p.n_chirps // 2) * p.velocity_resolution


def to_db(rd, eps=1e-12):
    """Magnitude in dB, normalized so the peak is 0 dB."""
    mag = np.abs(rd)
    return 20 * np.log10(mag / mag.max() + eps)