from dataclasses import dataclass

import numpy as np

from radar.params import RadarParams
from radar.waveform import beat_frequency, doppler_frequency


@dataclass(frozen=True)
class Target:
    range_m: float
    velocity_mps: float
    amplitude: float = 1.0


def simulate_frame(p, targets, snr_db=None, rng=None):
    """Return a complex (n_chirps, n_samples) beat-signal frame.

    snr_db is the per-sample SNR of a unit-amplitude target before any FFT.
    None means no noise.
    """
    n = np.arange(p.n_samples)[None, :]   # fast time index
    m = np.arange(p.n_chirps)[:, None]    # slow time index
    frame = np.zeros((p.n_chirps, p.n_samples), dtype=complex)

    for t in targets:
        f_b = beat_frequency(p, t.range_m)
        f_d = doppler_frequency(p, t.velocity_mps)
        phase = (2 * np.pi * (f_b * n / p.fs + f_d * m * p.chirp_time)
                 + 4 * np.pi * t.range_m / p.wavelength)
        frame += t.amplitude * np.exp(1j * phase)

    if snr_db is not None:
        rng = rng or np.random.default_rng()
        noise_power = 10 ** (-snr_db / 10)
        scale = np.sqrt(noise_power / 2)
        frame += scale * (rng.standard_normal(frame.shape)
                          + 1j * rng.standard_normal(frame.shape))
    return frame