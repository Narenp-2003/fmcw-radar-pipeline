import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter


def cfar_alpha(n_train, pfa):
    """CA-CFAR threshold factor for exponentially distributed noise power."""
    return n_train * (pfa ** (-1.0 / n_train) - 1.0)


def ca_cfar_2d(power, pfa=1e-3, guard=(2, 2), train=(4, 4)):
    """Boolean detection map, same shape as power.

    guard and train are (doppler, range) half-widths in cells.
    Doppler wraps around; range-edge cells without a full window are not flagged.
    """
    gd, gr = guard
    td, tr = train
    big = (2 * (gd + td) + 1, 2 * (gr + tr) + 1)
    small = (2 * gd + 1, 2 * gr + 1)
    n_big, n_small = big[0] * big[1], small[0] * small[1]
    n_train = n_big - n_small

    modes = ("wrap", "constant")  # axis 0 = Doppler, axis 1 = range
    sum_big = uniform_filter(power, size=big, mode=modes) * n_big
    sum_small = uniform_filter(power, size=small, mode=modes) * n_small
    noise = (sum_big - sum_small) / n_train

    det = power > cfar_alpha(n_train, pfa) * noise
    e = gr + tr
    det[:, :e] = False
    det[:, power.shape[1] - e:] = False
    return det


def extract_peaks(power, det):
    """(doppler_idx, range_idx) of detected cells that are 3x3 local maxima."""
    local_max = power == maximum_filter(power, size=3, mode=("wrap", "constant"))
    return np.argwhere(det & local_max)