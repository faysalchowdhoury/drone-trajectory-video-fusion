"""Render real NGSIM trajectories into an oblique 'drone' video with known ground truth.

The road plane (x lateral, y longitudinal, metres) is projected with a true homography that maps a
rectangle of road into a trapezoid, so far vehicles are smaller, as from a tilted camera. Each frame adds
asphalt texture, lane markings, per-vehicle grey levels, sensor noise and a 1-2 px hover jitter.
Ground truth (vehicle id, footprint centre in metres) is kept for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .geometry import apply, dlt

FT = 0.3048


@dataclass(frozen=True)
class Camera:
    width: int = 1280
    height: int = 720
    road_x: tuple[float, float] = (-1.0, 26.0)  # lateral extent (m)
    road_y: tuple[float, float] = (100.0, 340.0)  # longitudinal extent (m)

    def road_corners(self) -> np.ndarray:
        (x0, x1), (y0, y1) = self.road_x, self.road_y
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)

    def image_corners(self) -> np.ndarray:
        # near edge (y0) at the bottom, wide; far edge (y1) at the top, narrower: oblique view
        w, h = self.width, self.height
        return np.array([[0.30 * w, 0.97 * h], [0.70 * w, 0.97 * h], [0.58 * w, 0.03 * h], [0.42 * w, 0.03 * h]], float)

    def homography(self) -> np.ndarray:
        return dlt(self.road_corners(), self.image_corners())


def load_ngsim(raw_dir: str | Path, t0: float = 120.0, seconds: float = 60.0) -> pd.DataFrame:
    """Load NGSIM CSV pages (as written by cloud-trajectory-etl-geh-pipeline) and convert to SI."""
    files = sorted(Path(raw_dir).glob("page_*.csv"))
    if not files:
        raise FileNotFoundError(f"no NGSIM pages in {raw_dir}")
    cols = ["vehicle_id", "frame_id", "global_time", "local_x", "local_y", "v_length", "v_width", "lane_id"]
    df = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True).drop_duplicates(["vehicle_id", "frame_id"])
    df["t"] = (df["global_time"] - df["global_time"].min()) / 1000.0
    df = df[(df["t"] >= t0) & (df["t"] < t0 + seconds)].copy()
    df["x"] = df["local_x"] * FT
    df["y"] = df["local_y"] * FT
    df["length"] = df["v_length"] * FT
    df["width"] = df["v_width"] * FT
    df["frame"] = np.rint((df["t"] - t0) * 10).astype(int)
    return df[["frame", "vehicle_id", "x", "y", "length", "width", "lane_id"]].reset_index(drop=True)


def background(cam: Camera, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = np.full((cam.height, cam.width), 40, np.uint8)
    road = apply(h, cam.road_corners()).astype(np.int32)
    cv2.fillPoly(img, [road], 95)
    tex = rng.normal(0, 6, img.shape)
    img = np.clip(img + tex, 0, 255).astype(np.uint8)
    for lx in np.arange(0.0, 25.0, 3.7):  # lane markings
        for y in np.arange(cam.road_y[0], cam.road_y[1], 12.0):
            seg = apply(h, np.array([[lx, y], [lx, y + 3.0]])).astype(int)
            cv2.line(img, tuple(seg[0]), tuple(seg[1]), 170, 1)
    return img


def render(traj: pd.DataFrame, cam: Camera, *, seed: int = 0, noise_sd: float = 6.0, jitter_px: float = 1.5):  # type: ignore[no-untyped-def]
    """Yield ``(frame_index, image, ground_truth_rows)``."""
    rng = np.random.default_rng(seed)
    h = cam.homography()
    bg = background(cam, h, rng)
    shade = {vid: int(rng.choice([25, 35, 150, 190, 215, 235])) for vid in traj["vehicle_id"].unique()}
    in_view = (traj["y"] > cam.road_y[0]) & (traj["y"] < cam.road_y[1])
    for f, g in traj[in_view].groupby("frame"):
        img = bg.copy()
        for r in g.itertuples():
            # NGSIM local_y is the front bumper: footprint spans [y - length, y]
            poly = np.array([[r.x - r.width / 2, r.y - r.length], [r.x + r.width / 2, r.y - r.length],
                             [r.x + r.width / 2, r.y], [r.x - r.width / 2, r.y]])
            cv2.fillPoly(img, [apply(h, poly).astype(np.int32)], shade[r.vehicle_id])
        img = np.clip(img + rng.normal(0, noise_sd, img.shape), 0, 255).astype(np.uint8)
        jx, jy = rng.normal(0, jitter_px, 2)
        img = cv2.warpAffine(img, np.float32([[1, 0, jx], [0, 1, jy]]), (cam.width, cam.height), borderMode=cv2.BORDER_REPLICATE)
        gt = g.assign(cx=g["x"], cy=g["y"] - g["length"] / 2)[["vehicle_id", "cx", "cy"]]
        yield int(f), img, gt
