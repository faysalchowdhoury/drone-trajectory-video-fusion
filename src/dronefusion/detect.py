"""Vehicle detectors returning ``(boxes_xyxy, scores)`` per frame.

* :class:`BackgroundDetector`: median background over the first frames, phase-correlation
  stabilisation of each frame against it, absolute difference,
  Otsu-free fixed threshold, morphological open/close, connected components. Score = blob fill ratio
  times a size prior, so fragments and merged blobs get low scores, which is what ByteTrack's
  second association uses.
* :class:`YoloDetector`: adapter for Ultralytics YOLO (v8/v11) on real aerial footage. It is **not
  exercised in this repository** (no labelled footage or GPU here); install ``ultralytics`` and pass
  a weights file such as one fine-tuned on VisDrone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class BackgroundDetector:
    diff_thresh: int = 28
    min_area: int = 12
    bg_frames: list[np.ndarray] = field(default_factory=list)
    bg: np.ndarray | None = None

    def fit_background(self, frames: list[np.ndarray]) -> None:
        self.bg = np.median(np.stack(frames), axis=0).astype(np.uint8)

    def stabilise(self, img: np.ndarray) -> np.ndarray:
        """Undo the frame's global translation relative to the background (drone hover jitter).

        Without this, lane markings shifted by 1-2 px against the static background become false
        'vehicles' (103 false positives in a 3-vehicle test before this step was added).
        """
        assert self.bg is not None
        (dx, dy), _ = cv2.phaseCorrelate(np.float32(self.bg), np.float32(img))
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.bg is None:
            raise RuntimeError("call fit_background first")
        img = self.stabilise(img)
        d = cv2.absdiff(cv2.GaussianBlur(img, (3, 3), 0), cv2.GaussianBlur(self.bg, (3, 3), 0))
        _, m = cv2.threshold(d, self.diff_thresh, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        boxes, scores = [], []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < self.min_area:
                continue
            fill = area / max(w * h, 1)
            aspect = h / max(w, 1)
            size_prior = 1.0 if 0.6 <= aspect <= 6.0 else 0.4
            boxes.append([x, y, x + w, y + h])
            scores.append(float(np.clip(fill * size_prior, 0.0, 1.0)))
        return np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)


class YoloDetector:  # pragma: no cover - requires ultralytics weights and real footage
    def __init__(self, weights: str, conf: float = 0.1, classes: tuple[int, ...] = (2, 3, 5, 7)) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.classes = list(classes)

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.model.predict(img, conf=self.conf, classes=self.classes, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
