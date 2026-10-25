import numpy as np

from radar.params import RadarParams


def detections_to_measurements(p: RadarParams, peaks):
    """(doppler_idx, range_idx) peaks -> (n, 2) array of [range_m, velocity_mps]."""
    if len(peaks) == 0:
        return np.empty((0, 2))
    peaks = np.asarray(peaks)
    r = peaks[:, 1] * p.range_resolution
    v = (peaks[:, 0] - p.n_chirps // 2) * p.velocity_resolution
    return np.column_stack([r, v])


def measurement_noise(p: RadarParams):
    """Bin-quantisation noise: uniform error of one bin has sigma = res / sqrt(12)."""
    return np.diag([(p.range_resolution / np.sqrt(12)) ** 2,
                    (p.velocity_resolution / np.sqrt(12)) ** 2])