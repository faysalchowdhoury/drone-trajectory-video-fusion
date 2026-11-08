"""drone-trajectory-video-fusion.

Aerial-video trajectory extraction with ground truth: real NGSIM I-80 trajectories are rendered
into an oblique "drone" video (perspective camera, sensor noise), vehicles are detected (background
subtraction; a YOLO adapter is provided for real footage), tracked with a from-scratch ByteTrack,
georeferenced with a DLT homography from ground control points, scored against the true trajectories
(MOTA, ID switches, position RMSE), and fused with noisy loop-detector counts.
"""

__version__ = "1.0.0"
