import pytest
from radar.params import RadarParams

p = RadarParams()


def test_hand_values():
    assert p.wavelength == pytest.approx(3.90e-3, rel=1e-2)
    assert p.range_resolution == pytest.approx(0.15, rel=1e-2)
    assert p.max_range == pytest.approx(38.4, rel=1e-2)
    assert p.max_velocity == pytest.approx(24.3, rel=1e-2)
    assert p.velocity_resolution == pytest.approx(0.38, rel=1e-2)
    assert p.frame_time == pytest.approx(5.12e-3, rel=1e-6)


def test_consistency_identities():
    assert p.max_range == pytest.approx(p.n_samples * p.range_resolution)
    assert p.max_velocity == pytest.approx(p.n_chirps / 2 * p.velocity_resolution)