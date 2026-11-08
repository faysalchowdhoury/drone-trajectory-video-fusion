"""ByteTrack multi-object tracker (Zhang et al., ECCV 2022), from scratch.

Each track carries a constant-velocity Kalman filter on box state ``[cx, cy, w, h, vx, vy, vw, vh]``.
Per frame:

1. predict all tracks;
2. **first association**: high-score detections (``>= high_thresh``) vs all tracks, IoU cost, Hungarian;
3. **second association**: remaining tracks vs *low-score* detections (``low_thresh <= s < high_thresh``),
   which recovers occluded / partially visible objects that most trackers throw away;
4. unmatched high-score detections start new tracks; tracks unmatched for ``max_lost`` frames are removed.

Hungarian assignment uses ``scipy.optimize.linear_sum_assignment``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes ``a`` (N, 4) and ``b`` (M, 4) in ``x1, y1, x2, y2``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]


@dataclass
class _Track:
    tid: int
    x: np.ndarray  # state (8,)
    p: np.ndarray  # covariance (8, 8)
    lost: int = 0
    hits: int = 1

    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class ByteTracker:
    high_thresh: float = 0.6
    low_thresh: float = 0.1
    match_iou: float = 0.2
    max_lost: int = 30
    q: float = 1.0
    r: float = 4.0
    tracks: list[_Track] = field(default_factory=list)
    _next: int = 1

    def __post_init__(self) -> None:
        f = np.eye(8)
        f[:4, 4:] = np.eye(4)
        self.F = f
        self.H = np.eye(4, 8)
        self.Q = np.eye(8) * self.q
        self.Q[4:, 4:] *= 0.5
        self.R = np.eye(4) * self.r

    def _predict(self) -> None:
        for t in self.tracks:
            t.x = self.F @ t.x
            t.p = self.F @ t.p @ self.F.T + self.Q

    def _update(self, t: _Track, z: np.ndarray) -> None:
        y = z - self.H @ t.x
        s = self.H @ t.p @ self.H.T + self.R
        k = t.p @ self.H.T @ np.linalg.inv(s)
        t.x = t.x + k @ y
        t.p = (np.eye(8) - k @ self.H) @ t.p
        t.lost = 0
        t.hits += 1

    def _associate(self, tracks: list[_Track], boxes: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        m = iou(np.array([t.box() for t in tracks]), boxes)
        r, c = linear_sum_assignment(-m)
        pairs = [(i, j) for i, j in zip(r, c) if m[i, j] >= self.match_iou]
        mt = {i for i, _ in pairs}
        md = {j for _, j in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(boxes)) if j not in md]

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Process one frame; returns active ``(track_id, box)`` pairs updated this frame."""
        self._predict()
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        hi = scores >= self.high_thresh
        lo = (scores >= self.low_thresh) & ~hi
        z_hi, z_lo = xyxy_to_cxcywh(boxes[hi]), xyxy_to_cxcywh(boxes[lo])
        b_hi, b_lo = boxes[hi], boxes[lo]
        out: list[tuple[int, np.ndarray]] = []
        pairs, rem_t, rem_d = self._associate(self.tracks, b_hi)
        for i, j in pairs:
            self._update(self.tracks[i], z_hi[j])
            out.append((self.tracks[i].tid, self.tracks[i].box()))
        rest = [self.tracks[i] for i in rem_t]
        pairs2, rem_t2, _ = self._associate(rest, b_lo)
        for i, j in pairs2:
            self._update(rest[i], z_lo[j])
            out.append((rest[i].tid, rest[i].box()))
        for i in rem_t2:
            rest[i].lost += 1
        for j in rem_d:
            x = np.r_[z_hi[j], np.zeros(4)]
            p = np.diag([10, 10, 10, 10, 100, 100, 100, 100]).astype(float)
            t = _Track(self._next, x, p)
            self._next += 1
            self.tracks.append(t)
            out.append((t.tid, t.box()))
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        return out
