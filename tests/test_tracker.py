import numpy as np
import pytest

from radar.cfar import ca_cfar_2d, extract_peaks
from radar.channel import Target, simulate_frame
from radar.params import RadarParams
from radar.rdmap import range_doppler_map
from radar.tracker import Tracker
from radar.tracker_interface import detections_to_measurements, measurement_noise

p = RadarParams()
DT = p.frame_time
R = measurement_noise(p)


def truth(r0, v, k):
    return np.array([r0 + v * k * DT, v])


def test_measurement_conversion():
    z = detections_to_measurements(p, np.array([[p.n_chirps // 2 + 10, 60]]))
    assert z.shape == (1, 2)
    assert z[0, 0] == pytest.approx(60 * p.range_resolution)
    assert z[0, 1] == pytest.approx(10 * p.velocity_resolution)
    assert detections_to_measurements(p, np.empty((0, 2), dtype=int)).shape == (0, 2)


def test_confirms_after_three_hits():
    tr = Tracker(DT, R)
    sizes = [len(tr.step(truth(10.0, 5.0, k)[None, :])) for k in range(5)]
    assert sizes == [0, 0, 1, 1, 1]


def test_false_alarm_never_confirms():
    tr = Tracker(DT, R)
    assert tr.step(np.array([[12.0, 3.0]])) == []
    for _ in range(6):
        assert tr.step(np.empty((0, 2))) == []
    assert tr.tracks == []


def test_two_targets_two_tracks():
    tr = Tracker(DT, R)
    for k in range(30):
        z = np.vstack([truth(10.0, 5.0, k), truth(30.0, -8.0, k)])
        confirmed = tr.step(z)
    assert len(confirmed) == 2
    assert len({t.id for t in confirmed}) == 2
    for t in confirmed:
        err = min(np.abs(t.x - truth(10.0, 5.0, 29)).max(),
                  np.abs(t.x - truth(30.0, -8.0, 29)).max())
        assert err < 1e-6


def test_noisy_track_beats_raw_measurements():
    rng = np.random.default_rng(0)
    sigma = np.sqrt(np.diag(R))
    tr = Tracker(DT, R, gate_prob=1 - 1e-6)
    err_raw, err_trk = [], []
    for k in range(300):
        z = truth(10.0, 5.0, k) + rng.standard_normal(2) * sigma
        confirmed = tr.step(z[None, :])
        if k >= 50:
            assert len(confirmed) == 1
            err_raw.append(z - truth(10.0, 5.0, k))
            err_trk.append(confirmed[0].x - truth(10.0, 5.0, k))
    rms_raw = np.sqrt(np.mean(np.square(err_raw), axis=0))
    rms_trk = np.sqrt(np.mean(np.square(err_trk), axis=0))
    assert np.all(rms_trk < rms_raw)


def test_end_to_end_single_target():
    rng = np.random.default_rng(1)
    tr = Tracker(DT, R)
    r0, v0 = 20.0, 6.0
    for k in range(15):
        r, v = r0 + v0 * k * DT, v0
        frame = simulate_frame(p, [Target(r, v)], snr_db=-10.0, rng=rng)
        power = np.abs(range_doppler_map(frame)) ** 2
        z = detections_to_measurements(p, extract_peaks(power, ca_cfar_2d(power, 1e-6)))
        confirmed = tr.step(z)
    assert len(confirmed) == 1
    assert confirmed[0].x[0] == pytest.approx(r, abs=0.3)
    assert confirmed[0].x[1] == pytest.approx(v, abs=0.5)