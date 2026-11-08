"""Figures and Markdown report for a pipeline run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def write(res: dict[str, Any], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (out / "figures").mkdir(parents=True, exist_ok=True)
    if res.get("snapshot") is not None:
        img, boxes = res["snapshot"]
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for tid, b in boxes:
            x1, y1, x2, y2 = (int(v) for v in b)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (60, 200, 60), 1)
            cv2.putText(rgb, str(tid), (x1, max(y1 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (40, 220, 255), 1)
        cv2.imwrite(str(out / "figures" / "tracked_frame.png"), rgb)
    tr, gt = res["tracks"], res["gt"]
    lane = gt[(gt["cx"] > 7.0) & (gt["cx"] < 10.5)]  # one lane, time-space
    tl = tr[(tr["x"] > 7.0) & (tr["x"] < 10.5)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for v, g in lane.groupby("vehicle_id"):
        ax[0].plot(g["frame"] / 10, g["cy"], lw=0.7, color="#495057")
    for v, g in tl.groupby("tid"):
        ax[1].plot(g["frame"] / 10, g["y"], lw=0.7, color="#1c7ed6")
    ax[0].set_title("Ground truth (NGSIM), one lane")
    ax[1].set_title("Recovered from video (ByteTrack + homography)")
    for a in ax:
        a.set_xlabel("time (s)")
    ax[0].set_ylabel("distance along I-80 (m)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "time_space.png", dpi=130)
    plt.close(fig)
    summary = {k: v for k, v in res.items() if k not in ("tracks", "gt", "snapshot")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    m, c, cal = res["mot"], res["counts"], res["calibration"]
    L = ["# Drone pipeline results (NGSIM I-80, rendered oblique view)", "",
         f"{res['frames']} frames (10 Hz), {res['vehicles']} vehicles in view; {res['detections_per_frame']:.1f} detections per frame.", "",
         "| Stage | Result |", "|---|---|",
         f"| Homography from 4 GCPs (1 px click noise) | reprojection {cal['gcp_reprojection_px']:.2f} px; check-point error {cal['checkpoint_error_m']:.2f} m |",
         f"| MOTA | {m['mota']:.3f} |", f"| Precision / recall | {m['precision']:.3f} / {m['recall']:.3f} |",
         f"| ID switches | {m['id_switches']} |", f"| Position RMSE (matched, road coordinates) | {m['position_rmse_m']:.2f} m |",
         f"| Detection / tracking time | {res['detect_ms_per_frame']:.1f} / {res['track_ms_per_frame']:.1f} ms per frame |", "",
         f"## Count fusion at y = 220 m ({c['intervals']} intervals of {c['interval_s']:.0f} s; first half calibrates, second half tests)", "",
         f"Totals: truth {c['truth_total']:.0f}, video {c['video_total']:.0f}, loop {c['loop_total']:.0f}.", ""]
    if "rmse_fused" in c:
        L += ["| Estimate | RMSE per interval (veh) |", "|---|---|", f"| Loop detector | {c['rmse_loop']:.2f} |",
              f"| Video | {c['rmse_video']:.2f} |", f"| Fused (bias-corrected, inverse variance; video weight {c['video_weight']:.2f}) | {c['rmse_fused']:.2f} |"]
    L += ["", "![Tracked frame](figures/tracked_frame.png)", "", "![Time-space](figures/time_space.png)", ""]
    md = out / "RESULTS.md"
    md.write_text("\n".join(L), encoding="utf-8")
    return md
