from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dronefusion.api.app import create_app
from dronefusion.bytetrack import ByteTracker, iou
from dronefusion.detect import BackgroundDetector
from dronefusion.geometry import apply, dlt, reprojection_rmse
from dronefusion.pipeline import evaluate, fuse_counts
from dronefusion.render import Camera, render


def test_dlt_recovers_known_homography() -> None:
    h = np.array([[2.0, 0.3, 100.0], [0.1, 1.5, 50.0], [0.001, 0.0005, 1.0]])
    src = np.random.default_rng(0).uniform(0, 100, (10, 2))
    dst = apply(h, src)
    he = dlt(src, dst)
    assert np.allclose(he, h / h[2, 2], atol=1e-6)
    assert reprojection_rmse(he, src, dst) < 1e-6
    with pytest.raises(ValueError):
        dlt(src[:3], dst[:3])


def test_iou_and_tracker_keeps_identity() -> None:
    assert iou(np.array([[0, 0, 10, 10]]), np.array([[5, 0, 15, 10]]))[0, 0] == pytest.approx(50 / 150)
    t = ByteTracker()
    ids = []
    for k in range(30):
        boxes = np.array([[10 + 2 * k, 10, 30 + 2 * k, 20], [100, 50 + k, 110, 70 + k]], float)
        scores = np.array([0.9, 0.3 if k % 5 == 0 else 0.9])  # low-score detection every 5th frame
        out = t.update(boxes, scores)
        ids.append(sorted(tid for tid, _ in out))
    assert ids[-1] == [1, 2]  # both objects kept their ids through low-score frames


def _toy_traj() -> pd.DataFrame:
    rows = []
    for f in range(120):
        for vid, lane_x in ((1, 5.5), (2, 9.2), (3, 16.6)):
            rows.append((f, vid, lane_x, 150 + 1.2 * f + 15 * vid, 4.5, 1.8, 1))
    return pd.DataFrame(rows, columns=["frame", "vehicle_id", "x", "y", "length", "width", "lane_id"])


def test_render_detect_track_evaluate_end_to_end() -> None:
    cam = Camera()
    traj = _toy_traj()
    frames = list(render(traj, cam, seed=1))
    det = BackgroundDetector()
    det.fit_background([f[1] for f in frames[::6]])
    h_inv = np.linalg.inv(cam.homography())
    tr, rows, gts = ByteTracker(), [], []
    for f, img, gt in frames:
        b, s = det(img)
        for tid, box in tr.update(b, s):
            cx, cy = apply(h_inv, np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]))[0]
            rows.append((f, tid, cx, cy))
        gts.append(gt.assign(frame=f))
    m = evaluate(pd.DataFrame(rows, columns=["frame", "tid", "x", "y"]), pd.concat(gts))
    assert m["mota"] > 0.8 and m["position_rmse_m"] < 1.5 and m["id_switches"] <= 3


def test_fusion_beats_loop_alone_on_biased_loop() -> None:
    rng = np.random.default_rng(0)
    rows_gt, rows_tr = [], []
    for v in range(400):
        f0 = int(rng.integers(0, 3000))
        for k in range(30):
            rows_gt.append((f0 + k, v, 0.0, 205 + k))
            rows_tr.append((f0 + k, v, 0.0, 205 + k + rng.normal(0, 0.2)))
    gt = pd.DataFrame(rows_gt, columns=["frame", "vehicle_id", "cx", "cy"])
    tr = pd.DataFrame(rows_tr, columns=["frame", "tid", "x", "y"])
    c = fuse_counts(tr, gt, rng)
    assert c["rmse_fused"] <= c["rmse_loop"] + 1e-9


def test_api() -> None:
    cl = TestClient(create_app())
    cam = Camera()
    r = cl.post("/v1/homography", json={"road_m": cam.road_corners().tolist(), "pixels": cam.image_corners().tolist()})
    assert r.status_code == 200 and r.json()["reprojection_rmse_px"] < 1e-6
    g = cl.post("/v1/georeference", json={"h": r.json()["h"], "pixels": [cam.image_corners()[0].tolist()]})
    assert np.allclose(g.json()["road_m"][0], cam.road_corners()[0], atol=1e-6)
    assert cl.post("/v1/georeference", json={"h": [[0, 0, 0], [0, 0, 0], [0, 0, 0]], "pixels": [[1, 1]]}).status_code == 422
