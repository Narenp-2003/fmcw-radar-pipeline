import matplotlib.pyplot as plt
import numpy as np

from radar.channel import Target, simulate_frame
from radar.params import RadarParams
from radar.rdmap import range_axis, range_doppler_map, to_db, velocity_axis

p = RadarParams()
targets = [Target(10.0, 5.0), Target(20.0, -8.0), Target(28.0, 12.0, 0.7)]

rng = np.random.default_rng(0)
rd = range_doppler_map(simulate_frame(p, targets, snr_db=20, rng=rng))
db = to_db(rd)

r, v = range_axis(p), velocity_axis(p)
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(db, origin="lower", aspect="auto", vmin=-70, vmax=0,
               extent=[r[0], r[-1], v[0], v[-1]])
ax.plot([t.range_m for t in targets], [t.velocity_mps for t in targets],
        "rx", markersize=10, label="true targets")
ax.set_xlabel("Range (m)")
ax.set_ylabel("Velocity (m/s)")
ax.set_title("Range-Doppler map (SNR 20 dB per sample)")
ax.legend(loc="upper right")
fig.colorbar(im, label="dB (relative to peak)")
fig.tight_layout()
fig.savefig("results/rd_map.png", dpi=150)
plt.show()