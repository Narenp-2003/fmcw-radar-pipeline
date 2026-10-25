import matplotlib.pyplot as plt
import numpy as np

from radar.cfar import ca_cfar_2d, os_cfar_2d
from radar.channel import Target, simulate_frame
from radar.params import RadarParams
from radar.rdmap import range_doppler_map

p = RadarParams()
M, RES, VRES = p.n_chirps, p.range_resolution, p.velocity_resolution
GUARD, TRAIN = (2, 2), (4, 4)
PFA, TRIALS = 1e-4, 100
LO, HI = 40, 100                    # evaluate on a range crop to save time
CUT = (M // 2 + 10, 60 - LO)        # target cell (doppler idx, range idx in crop)
snrs = np.arange(-36, -23, 1.5)

target = Target(60 * RES, 10 * VRES)
interferer = Target(64 * RES, 10 * VRES, amplitude=30.0)  # 4 bins away, +29.5 dB


def run(targets):
    rng = np.random.default_rng(0)
    pd_ca, pd_os = [], []
    for snr in snrs:
        h_ca = h_os = 0
        for _ in range(TRIALS):
            frame = simulate_frame(p, targets, snr_db=snr, rng=rng)
            crop = (np.abs(range_doppler_map(frame)) ** 2)[:, LO:HI]
            h_ca += ca_cfar_2d(crop, PFA, GUARD, TRAIN)[CUT]
            h_os += os_cfar_2d(crop, PFA, GUARD, TRAIN)[CUT]
        pd_ca.append(h_ca / TRIALS)
        pd_os.append(h_os / TRIALS)
    return pd_ca, pd_os


fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
cases = ((axes[0], [target], "Single target"),
         (axes[1], [target, interferer], "Strong interferer 4 bins away"))
for ax, tgts, title in cases:
    pd_ca, pd_os = run(tgts)
    ax.plot(snrs, pd_ca, "o-", label="CA-CFAR")
    ax.plot(snrs, pd_os, "s-", label="OS-CFAR (k = 0.75 N)")
    ax.set_title(title)
    ax.set_xlabel("Per-sample SNR of target (dB)")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Probability of detection")
axes[0].legend()
fig.suptitle(f"Pd vs SNR (design Pfa = {PFA:g}, {TRIALS} trials per point)")
fig.tight_layout()
fig.savefig("results/pd_vs_snr.png", dpi=150)
plt.show()