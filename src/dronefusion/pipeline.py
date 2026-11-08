"""End-to-end run: render -> detect -> ByteTrack -> georeference -> evaluate -> fuse counts.

Evaluation (CLEAR-MOT style, in road coordinates): each frame, tracks are matched to ground-truth
vehicles by Hungarian assignment on distance with a 3-m gate. MOTA = 1 - (FN + FP + IDSW) / GT;
position RMSE is over matched pairs.

Count fusion at a virtual detector line (y = 220 m): ground truth counts per 10-s interval; a "loop"
estimate with a systematic 6% undercount plus N(0, 1.5) noise; a "video" estimate from track
crossings. Bias and variance of each sensor are estimated on the first half of the intervals, and the
second half is fused by bias-corrected inverse-variance weighting and scored against truth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .bytetrack import ByteTracker
from .detect import BackgroundDetector
from .geometry import apply, dlt, reprojection_rmse
from .render import Camera, load_ngsim, render

LINE_Y = 220.0


@dataclass(frozen=True)
class RunConfig:
    raw_dir: Path
    t0: float = 120.0
    seconds: float = 300.0
    gcp_click_noise_px: float = 1.0
    seed: int = 0


def calibrate(cam: Camera, rng: np.random.Generator, click_noise: float) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate the homography from 6 surveyed ground control points (lane-marking ends), clicked with pixel noise."""
    h_true = cam.homography()
    gcp = np.array([[0.0, 120.0], [22.2, 120.0], [0.0, 320.0], [22.2, 320.0], [11.1, 200.0], [3.7, 260.0]])
    px = apply(h_true, gcp) + rng.normal(0, click_noise, (len(gcp), 2))
    h_est = dlt(gcp[:4], px[:4])
    check = np.array([[7.4, 150.0], [18.5, 300.0], [14.8, 230.0]])
    return h_est, {"gcp_reprojection_px": reprojection_rmse(h_est, gcp, px),
                   "checkpoint_error_m": float(np.sqrt(np.mean(np.sum((apply(np.linalg.inv(h_est), apply(h_true, check)) - check) ** 2, axis=1))))}


def run(cfg: RunConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    cam = Camera()
    traj = load_ngsim(cfg.raw_dir, cfg.t0, cfg.seconds)
    h_est, calib = calibrate(cam, rng, cfg.gcp_click_noise_px)
    h_inv = np.linalg.inv(h_est)
    # pass 1: sample frames for the background model (rendering is deterministic per seed, so
    # pass 2 regenerates the same video; nothing is held in memory: 5 min would be ~2.8 GB)
    det = BackgroundDetector()
    sample = []
    for k, (_, img, _) in enumerate(render(traj, cam, seed=cfg.seed)):
        if k % 10 == 0:
            sample.append(img)
        if len(sample) >= 30:
            break
    det.fit_background(sample)
    tracker = ByteTracker()
    t_det = t_trk = 0.0
    rows, gt_rows = [], []
    n_det = 0
    n_frames = 0
    snapshot = None
    for f, img, gt in render(traj, cam, seed=cfg.seed):
        n_frames += 1
        s = time.perf_counter()
        boxes, scores = det(img)
        t_det += time.perf_counter() - s
        n_det += len(boxes)
        s = time.perf_counter()
        out = tracker.update(boxes, scores)
        t_trk += time.perf_counter() - s
        for tid, b in out:
            cx, cy = apply(h_inv, np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]]))[0]
            rows.append((f, tid, cx, cy))
        gt_rows.append(gt.assign(frame=f))
        if n_frames == 1200:
            snapshot = (img.copy(), [(tid, b.copy()) for tid, b in out])
    tracks = pd.DataFrame(rows, columns=["frame", "tid", "x", "y"])
    gt_all = pd.concat(gt_rows, ignore_index=True)
    mot = evaluate(tracks, gt_all)
    counts = fuse_counts(tracks, gt_all, rng)
    return {"frames": n_frames, "vehicles": int(gt_all["vehicle_id"].nunique()), "detections_per_frame": n_det / n_frames,
            "detect_ms_per_frame": 1000 * t_det / n_frames, "track_ms_per_frame": 1000 * t_trk / n_frames,
            "calibration": calib, "mot": mot, "counts": counts, "tracks": tracks, "gt": gt_all, "snapshot": snapshot}


def evaluate(tracks: pd.DataFrame, gt: pd.DataFrame, gate_m: float = 3.0) -> dict[str, float]:
    tp = fp = fn = idsw = 0
    sq = []
    last: dict[int, int] = {}
    for f, g in gt.groupby("frame"):
        tr = tracks[tracks["frame"] == f]
        a = g[["cx", "cy"]].to_numpy()
        b = tr[["x", "y"]].to_numpy()
        if len(a) and len(b):
            d = np.linalg.norm(a[:, None] - b[None], axis=2)
            r, c = linear_sum_assignment(d)
            ok = d[r, c] <= gate_m
            r, c = r[ok], c[ok]
        else:
            r = c = np.empty(0, int)
        tp += len(r)
        fn += len(a) - len(r)
        fp += len(b) - len(c)
        gids = g["vehicle_id"].to_numpy()
        tids = tr["tid"].to_numpy()
        for i, j in zip(r, c):
            sq.append(float(np.sum((a[i] - b[j]) ** 2)))
            prev = last.get(int(gids[i]))
            if prev is not None and prev != int(tids[j]):
                idsw += 1
            last[int(gids[i])] = int(tids[j])
    n_gt = len(gt)
    return {"gt_objects": n_gt, "tp": tp, "fp": fp, "fn": fn, "id_switches": idsw,
            "mota": 1.0 - (fn + fp + idsw) / max(n_gt, 1), "recall": tp / max(n_gt, 1), "precision": tp / max(tp + fp, 1),
            "position_rmse_m": float(np.sqrt(np.mean(sq))) if sq else float("nan")}


def _crossings(df: pd.DataFrame, id_col: str, y_col: str) -> np.ndarray:
    out = []
    for _, g in df.sort_values("frame").groupby(id_col):
        y = g[y_col].to_numpy()
        fr = g["frame"].to_numpy()
        k = np.flatnonzero((y[:-1] < LINE_Y) & (y[1:] >= LINE_Y))
        if k.size:
            out.append(fr[k[0] + 1])
    return np.asarray(out, int)


def fuse_counts(tracks: pd.DataFrame, gt: pd.DataFrame, rng: np.random.Generator, interval_frames: int = 100) -> dict[str, Any]:
    n_frames = int(gt["frame"].max()) + 1
    edges = np.arange(0, n_frames + interval_frames, interval_frames)
    truth = np.histogram(_crossings(gt, "vehicle_id", "cy"), edges)[0].astype(float)
    video = np.histogram(_crossings(tracks, "tid", "y"), edges)[0].astype(float)
    loop = np.maximum(np.round(truth * 0.94 + rng.normal(0, 1.5, truth.size)), 0)
    half = truth.size // 2
    out: dict[str, Any] = {"intervals": int(truth.size), "interval_s": interval_frames / 10.0,
                           "truth_total": float(truth.sum()), "video_total": float(video.sum()), "loop_total": float(loop.sum())}
    if half < 2:
        return out
    bias_v, bias_l = (video[:half] - truth[:half]).mean(), (loop[:half] - truth[:half]).mean()
    var_v = max(np.var(video[:half] - truth[:half], ddof=1), 0.25)
    var_l = max(np.var(loop[:half] - truth[:half], ddof=1), 0.25)
    w_v = (1 / var_v) / (1 / var_v + 1 / var_l)
    fused = w_v * (video[half:] - bias_v) + (1 - w_v) * (loop[half:] - bias_l)
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))  # noqa: E731
    out.update({"video_weight": float(w_v), "rmse_loop": rmse(loop[half:] - truth[half:]), "rmse_video": rmse(video[half:] - truth[half:]),
                "rmse_fused": rmse(fused - truth[half:])})
    return out
