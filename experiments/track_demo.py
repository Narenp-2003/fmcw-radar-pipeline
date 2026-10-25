import matplotlib.pyplot as plt
import numpy as np

from radar.cfar import ca_cfar_2d, extract_peaks
from radar.channel import Target, simulate_frame
from radar.params import RadarParams
from radar.rdmap import range_doppler_map
from radar.tracker import Tracker
from radar.tracker_interface import detections_to_measurements, measurement_noise

p = RadarParams()
SNR_DB, PFA, N_FRAMES = -10.0, 1e-5, 300
rng = np.random.default_rng(0)

# (range0 [m], velocity0 [m/s], acceleration [m/s^2]); the third target manoeuvres
SCENARIO = [(10.0, 5.0, 0.0), (32.0, -8.0, 0.0), (22.0, 10.0, -12.0)]


def truth_at(t):
    return [(r0 + v0 * t + 0.5 * a * t ** 2, v0 + a * t) for r0, v0, a in SCENARIO]


def rms(e, col):
    return float(np.sqrt(np.mean(np.asarray(e)[:, col] ** 2)))


tracker = Tracker(dt=p.frame_time, R=measurement_noise(p), q=1.0)
raw, tracks_log, truth_log = [], [], []

for k in range(N_FRAMES):
    truth = truth_at(k * p.frame_time)
    frame = simulate_frame(p, [Target(r, v) for r, v in truth], snr_db=SNR_DB, rng=rng)
    power = np.abs(range_doppler_map(frame)) ** 2
    z = detections_to_measurements(p, extract_peaks(power, ca_cfar_2d(power, PFA)))
    confirmed = tracker.step(z)
    raw.append(z)
    tracks_log.append([(tr.id, tr.x.copy()) for tr in confirmed])
    truth_log.append(truth)

# ---- metrics ----
err_raw = [[] for _ in SCENARIO]
err_trk = [[] for _ in SCENARIO]
tracked = np.zeros(len(SCENARIO))
false_ids = set()
for z, trk, truth in zip(raw, tracks_log, truth_log):
    for i, (r, v) in enumerate(truth):
        if len(z):
            j = np.argmin(np.abs(z[:, 0] - r))
            if abs(z[j, 0] - r) < 1.0:
                err_raw[i].append(z[j] - [r, v])
        if trk:
            states = np.array([x for _, x in trk])
            j = np.argmin(np.abs(states[:, 0] - r))
            if abs(states[j, 0] - r) < 1.0:
                err_trk[i].append(states[j] - [r, v])
                tracked[i] += 1
    for tid, x in trk:
        if all(abs(x[0] - r) >= 1.0 for r, _ in truth):
            false_ids.add(tid)

for i in range(len(SCENARIO)):
    print(f"target {i + 1}: tracked {tracked[i] / N_FRAMES:6.1%} | "
          f"range RMSE raw {rms(err_raw[i], 0):.3f} -> track {rms(err_trk[i], 0):.3f} m | "
          f"velocity RMSE raw {rms(err_raw[i], 1):.3f} -> track {rms(err_trk[i], 1):.3f} m/s")
print("false confirmed tracks:", len(false_ids))

# ---- plot ----
t = np.arange(N_FRAMES) * p.frame_time
raw_t = np.concatenate([np.full(len(z), tk) for z, tk in zip(raw, t)])
raw_all = np.vstack(raw)
truth_arr = np.array(truth_log)          # (frames, targets, 2)

paths = {}
for k, trk in enumerate(tracks_log):
    for tid, x in trk:
        paths.setdefault(tid, []).append((t[k], x[0], x[1]))

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for col, ax, label in ((0, axes[0], "Range (m)"), (1, axes[1], "Velocity (m/s)")):
    ax.plot(raw_t, raw_all[:, col], ".", color="0.75", markersize=3, label="CFAR detections")
    for i in range(len(SCENARIO)):
        ax.plot(t, truth_arr[:, i, col], "k--", lw=1, label="truth" if i == 0 else None)
    for n, (tid, pts) in enumerate(paths.items()):
        a = np.array(pts)
        ax.plot(a[:, 0], a[:, 1 + col], lw=2, label="confirmed track" if n == 0 else None)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
axes[0].legend(loc="upper right")
axes[1].set_xlabel("Time (s)")
fig.suptitle(f"Radar tracking (SNR {SNR_DB:g} dB per sample, Pfa {PFA:g})")
fig.tight_layout()
fig.savefig("results/tracking.png", dpi=150)
plt.show()