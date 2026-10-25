import numpy as np
from scipy.ndimage import maximum_filter, rank_filter, uniform_filter
from scipy.optimize import brentq


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

def os_cfar_alpha(n_train, k, pfa):
    """OS-CFAR threshold factor: Pfa = prod_{j=N-k+1}^{N} j / (j + alpha)."""
    j = np.arange(n_train - k + 1, n_train + 1, dtype=float)

    def excess(alpha):
        return np.sum(np.log(j / (j + alpha))) - np.log(pfa)

    return brentq(excess, 0.0, 1e12)


def os_cfar_2d(power, pfa=1e-3, guard=(2, 2), train=(4, 4), k=None):
    """Ordered-statistic CFAR: threshold = alpha * (k-th smallest training cell).

    k defaults to 0.75 * number of training cells.
    """
    gd, gr = guard
    td, tr = train
    big = (2 * (gd + td) + 1, 2 * (gr + tr) + 1)
    footprint = np.ones(big, dtype=bool)
    footprint[td:td + 2 * gd + 1, tr:tr + 2 * gr + 1] = False  # guard + CUT
    n_train = int(footprint.sum())
    k = k or int(0.75 * n_train)

    # Doppler wraps (pad by wrapping); range edges are masked out below.
    pad = gd + td
    n_dop = power.shape[0]
    padded = np.pad(power, ((pad, pad), (0, 0)), mode="wrap")
    noise = rank_filter(padded, k - 1, footprint=footprint, mode="constant")
    noise = noise[pad:pad + n_dop, :]

    det = power > os_cfar_alpha(n_train, k, pfa) * noise
    e = gr + tr
    det[:, :e] = False
    det[:, power.shape[1] - e:] = False
    return det