import numpy as np
import pytest

from radar.cfar import ca_cfar_2d, cfar_alpha, extract_peaks, os_cfar_2d, os_cfar_alpha
from radar.channel import Target, simulate_frame
from radar.params import RadarParams
from radar.rdmap import range_doppler_map

p = RadarParams()
M = p.n_chirps
RES, VRES = p.range_resolution, p.velocity_resolution
GUARD, TRAIN = (2, 2), (4, 4)
EDGE = GUARD[1] + TRAIN[1]


def targets():
    return [Target(60 * RES, 10 * VRES), Target(80 * RES, -20 * VRES)]


def power_map(tgts, snr_db, seed, window=True):
    rng = np.random.default_rng(seed)
    frame = simulate_frame(p, tgts, snr_db=snr_db, rng=rng)
    return np.abs(range_doppler_map(frame, window=window)) ** 2


def test_alpha_hand_value():
    assert cfar_alpha(144, 1e-3) == pytest.approx(7.08, rel=1e-2)


def test_detects_targets():
    det = ca_cfar_2d(power_map(targets(), 10.0, 0), pfa=1e-4, guard=GUARD, train=TRAIN)
    assert det[M // 2 + 10, 60]
    assert det[M // 2 - 20, 80]


def test_range_edges_not_flagged():
    det = ca_cfar_2d(power_map([], 10.0, 1), pfa=0.5, guard=GUARD, train=TRAIN)
    assert det.any()
    assert not det[:, :EDGE].any()
    assert not det[:, p.n_samples - EDGE:].any()


def test_pfa_matches_design():
    pfa, frames = 1e-3, 40
    cells = M * (p.n_samples - 2 * EDGE)
    hits = sum(
        ca_cfar_2d(power_map([], 10.0, s, window=False), pfa, GUARD, TRAIN).sum()
        for s in range(frames)
    )
    assert hits / (frames * cells) == pytest.approx(pfa, rel=0.3)


def test_extract_peaks_contains_targets():
    power = power_map(targets(), 10.0, 2)
    det = ca_cfar_2d(power, pfa=1e-4, guard=GUARD, train=TRAIN)
    found = {tuple(x) for x in extract_peaks(power, det)}
    assert {(M // 2 + 10, 60), (M // 2 - 20, 80)} <= found


def test_os_alpha_k1_hand_value():
    assert os_cfar_alpha(144, 1, 1e-3) == pytest.approx(144 * (1 / 1e-3 - 1), rel=1e-6)


def test_os_detects_targets():
    det = os_cfar_2d(power_map(targets(), 10.0, 0), pfa=1e-4, guard=GUARD, train=TRAIN)
    assert det[M // 2 + 10, 60]
    assert det[M // 2 - 20, 80]


def test_os_pfa_matches_design():
    pfa, frames = 1e-3, 40
    cells = M * (p.n_samples - 2 * EDGE)
    hits = sum(
        os_cfar_2d(power_map([], 10.0, s, window=False), pfa, GUARD, TRAIN).sum()
        for s in range(frames)
    )
    assert hits / (frames * cells) == pytest.approx(pfa, rel=0.35)


def test_os_survives_masking_interferer():
    tgts = [Target(60 * RES, 10 * VRES), Target(64 * RES, 10 * VRES, amplitude=30.0)]
    power = power_map(tgts, -20.0, 3)
    cut = (M // 2 + 10, 60)
    assert not ca_cfar_2d(power, 1e-4, GUARD, TRAIN)[cut]
    assert os_cfar_2d(power, 1e-4, GUARD, TRAIN)[cut]