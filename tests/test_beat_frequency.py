import numpy as np
import pytest

from radar.channel import Target, simulate_frame
from radar.params import RadarParams
from radar.waveform import beat_frequency

p = RadarParams()


def test_beat_frequency_formula():
    assert beat_frequency(p, 10.0) == pytest.approx(2 * p.slope * 10.0 / 3e8, rel=1e-3)


def test_max_range_beat_equals_fs():
    assert beat_frequency(p, p.max_range) == pytest.approx(p.fs)


def test_range_peak_bin():
    k = 60
    frame = simulate_frame(p, [Target(range_m=k * p.range_resolution, velocity_mps=0.0)])
    spectrum = np.abs(np.fft.fft(frame[0]))
    assert int(np.argmax(spectrum)) == k


def test_doppler_peak_bin():
    k = 10
    frame = simulate_frame(p, [Target(range_m=60 * p.range_resolution,
                                      velocity_mps=k * p.velocity_resolution)])
    slow = np.abs(np.fft.fft(frame[:, 60]))
    assert int(np.argmax(slow)) == k


def test_noise_power():
    rng = np.random.default_rng(0)
    frame = simulate_frame(p, [], snr_db=10.0, rng=rng)
    assert np.mean(np.abs(frame) ** 2) == pytest.approx(0.1, rel=0.05)