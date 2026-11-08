"""Planar homography (normalised DLT) between road coordinates (metres) and image pixels.

``H`` maps homogeneous road points ``[x, y, 1]`` to pixels ``[u, v, 1]`` up to scale. It is estimated
from >= 4 ground control points with Hartley normalisation (centroid at the origin, mean distance
sqrt(2)) and SVD; its inverse georeferences detections. Reprojection RMSE on held-out control points
measures calibration quality.
"""

from __future__ import annotations

import numpy as np


def _normalise(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = p.mean(axis=0)
    d = np.sqrt(((p - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / max(d, 1e-12)
    t = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    ph = np.c_[p, np.ones(len(p))] @ t.T
    return ph, t


def dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Homography mapping ``src`` (N, 2) to ``dst`` (N, 2), N >= 4."""
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError("need >= 4 matching point pairs")
    a, ta = _normalise(np.asarray(src, float))
    b, tb = _normalise(np.asarray(dst, float))
    rows = []
    for (x, y, _), (u, v, _) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    hn = vt[-1].reshape(3, 3)
    h = np.linalg.inv(tb) @ hn @ ta
    return h / h[2, 2]


def apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3]


def reprojection_rmse(h: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((apply(h, src) - dst) ** 2, axis=1))))
