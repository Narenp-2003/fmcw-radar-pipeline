from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2


@dataclass
class Track:
    id: int
    x: np.ndarray            # [range, velocity]
    P: np.ndarray            # 2x2 covariance
    hits: int = 1
    age: int = 1             # frames since birth, including the birth frame
    misses: int = 0          # consecutive misses
    confirmed: bool = False


class Tracker:
    """Constant-velocity Kalman tracker with gating, GNN association, M-of-N confirmation."""

    def __init__(self, dt, R, q=1.0, gate_prob=0.999, confirm=(3, 5), max_misses=5):
        self.R = np.asarray(R, dtype=float)
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        self.gate = chi2.ppf(gate_prob, df=2)
        self.M, self.N = confirm
        self.max_misses = max_misses
        self.tracks = []
        self._next_id = 0

    def step(self, z):
        """Advance one frame with measurements z (n, 2). Returns the confirmed tracks."""
        z = np.asarray(z, dtype=float).reshape(-1, 2)

        # 1. Predict
        for t in self.tracks:
            t.x = self.F @ t.x
            t.P = self.F @ t.P @ self.F.T + self.Q
            t.age += 1

        # 2. Associate: gated Mahalanobis cost, Hungarian assignment
        matched_t, matched_z = set(), set()
        if self.tracks and len(z):
            big = 1e9
            cost = np.full((len(self.tracks), len(z)), big)
            for i, t in enumerate(self.tracks):
                s_inv = np.linalg.inv(t.P + self.R)
                for j in range(len(z)):
                    y = z[j] - t.x
                    d2 = y @ s_inv @ y
                    if d2 < self.gate:
                        cost[i, j] = d2
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < self.gate:
                    self._update(self.tracks[i], z[j])
                    matched_t.add(i)
                    matched_z.add(j)

        # 3. Track management (existing tracks only)
        survivors = []
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
            if not t.confirmed:
                if t.hits >= self.M:
                    t.confirmed = True
                elif t.hits + (self.N - t.age) < self.M:
                    continue                      # can no longer confirm: drop
            elif t.misses > self.max_misses:
                continue                          # confirmed track lost: drop
            survivors.append(t)
        self.tracks = survivors

        # 4. Start new tentative tracks from unmatched measurements
        for j in range(len(z)):
            if j not in matched_z:
                self.tracks.append(Track(self._next_id, z[j].copy(), self.R.copy()))
                self._next_id += 1

        return [t for t in self.tracks if t.confirmed]

    def _update(self, t, z):
        s = t.P + self.R
        k = t.P @ np.linalg.inv(s)
        t.x = t.x + k @ (z - t.x)
        i_k = np.eye(2) - k
        t.P = i_k @ t.P @ i_k.T + k @ self.R @ k.T   # Joseph form
        t.hits += 1
        t.misses = 0