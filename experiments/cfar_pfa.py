import numpy as np

from radar.cfar import ca_cfar_2d
from radar.channel import simulate_frame
from radar.params import RadarParams
from radar.rdmap import range_doppler_map

p = RadarParams()
rng = np.random.default_rng(1)
GUARD, TRAIN = (2, 2), (4, 4)
EDGE = GUARD[1] + TRAIN[1]
cells = p.n_chirps * (p.n_samples - 2 * EDGE)
frames = 100

for pfa in (1e-2, 1e-3, 1e-4):
    for window in (False, True):
        hits = 0
        for _ in range(frames):
            frame = simulate_frame(p, [], snr_db=10, rng=rng)
            power = np.abs(range_doppler_map(frame, window=window)) ** 2
            hits += ca_cfar_2d(power, pfa, GUARD, TRAIN).sum()
        print(f"design {pfa:.0e}  hann={window!s:5}  measured {hits / (frames * cells):.2e}")