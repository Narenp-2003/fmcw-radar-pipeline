import matplotlib.pyplot as plt
import numpy as np

from radar.cfar import ca_cfar_2d, extract_peaks
from radar.channel import Target, simulate_frame
from radar.params import RadarParams
from radar.rdmap import range_axis, range_doppler_map, to_db, velocity_axis

p = RadarParams()
targets = [Target(10.0, 5.0), Target(20.0, -8.0), Target(28.0, 12.0, 0.7)]
rng = np.random.default_rng(0)

rd = range_doppler_map(simulate_frame(p, targets, snr_db=-10, rng=rng))
power = np.abs(rd) ** 2
peaks = extract_peaks(power, ca_cfar_2d(power, pfa=1e-5))

r, v = range_axis(p), velocity_axis(p)
for di, ri in peaks:
    print(f"range {r[ri]:.2f} m, velocity {v[di]:.2f} m/s")

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(to_db(rd), origin="lower", aspect="auto", vmin=-45, vmax=0,
               extent=[r[0], r[-1], v[0], v[-1]])
ax.plot([t.range_m for t in targets], [t.velocity_mps for t in targets],
        "rx", markersize=10, label="true targets")
ax.plot(r[peaks[:, 1]], v[peaks[:, 0]], "wo", markerfacecolor="none",
        markersize=14, label="CFAR peaks")
ax.set_xlabel("Range (m)")
ax.set_ylabel("Velocity (m/s)")
ax.set_title("CA-CFAR detections (SNR -10 dB per sample, Pfa 1e-5)")
ax.legend(loc="upper right")
fig.colorbar(im, label="dB (relative to peak)")
fig.tight_layout()
fig.savefig("results/cfar_detections.png", dpi=150)
plt.show()