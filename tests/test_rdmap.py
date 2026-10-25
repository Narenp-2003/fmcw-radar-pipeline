import numpy as np
import pytest

from radar.channel import Target, simulate_frame
from radar.params import RadarParams
from radar.rdmap import range_axis, range_doppler_map, velocity_axis

p = RadarParams()
M, N = p.n_chirps, p.n_samples
RES, VRES = p.range_resolution, p.velocity_resolution


def peak_index(rd):
    return np.unravel_index(np.argmax(np.abs(rd)), rd.shape)


def test_single_target_peak():
    rd = range_doppler_map(simulate_frame(p, [Target(60 * RES, 10 * VRES)]))
    assert peak_index(rd) == (M // 2 + 10, 60)


def test_negative_velocity():
    rd = range_doppler_map(simulate_frame(p, [Target(60 * RES, -10 * VRES)]))
    assert peak_index(rd) == (M // 2 - 10, 60)


def test_off_bin_target_within_one_bin():
    r = 60.5 * RES
    rd = range_doppler_map(simulate_frame(p, [Target(r, 0.0)]))
    _, ri = peak_index(rd)
    assert abs(range_axis(p)[ri] - r) <= RES


def test_two_targets_resolved():
    t1 = Target(60 * RES, 10 * VRES)
    t2 = Target(80 * RES, -20 * VRES)
    mag = np.abs(range_doppler_map(simulate_frame(p, [t1, t2])))
    top2 = {tuple(np.unravel_index(i, mag.shape)) for i in np.argsort(mag.ravel())[-2:]}
    assert top2 == {(M // 2 + 10, 60), (M // 2 - 20, 80)}


def test_axes():
    r, v = range_axis(p), velocity_axis(p)
    assert r[1] - r[0] == pytest.approx(RES)
    assert r[-1] == pytest.approx(p.max_range - RES)
    assert v[M // 2] == 0
    assert v[M // 2 + 10] == pytest.approx(10 * VRES)