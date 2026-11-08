I combined two traffic counters and the result was worse than the better one alone.

I built a drone-video pipeline that turns pixels into georeferenced vehicle trajectories: homography from ground control points, frame stabilisation, vehicle detection, a from-scratch ByteTrack tracker, and CLEAR-MOT scoring. To get exact ground truth, I rendered real NGSIM I-80 trajectories into an oblique drone view with hover jitter and noise, then gave the pipeline only the pixels.

Over 5 minutes and 766 vehicles:
🔹 MOTA 0.942, precision 0.998, recall 0.944
🔹 Position error 0.30 m in road coordinates; georeferencing check points 0.62 m
🔹 Video counts beat the simulated loop detector: 0.89 vs 1.91 vehicles RMSE per 10 s

Then I fused the two with textbook inverse-variance weighting, and the fused count (1.24) was worse than video alone. With only 15 calibration intervals, the variance estimates were noisy enough to give the better sensor a weight of 0.11. The formula is optimal when you know the variances. When you estimate them from a few minutes of data, it isn't.

Two more lessons:
1️⃣ Stabilise before you subtract. A 1-2 px hover drift turned lane markings into 103 phantom vehicles in a 3-car test.
2️⃣ Recall limits these pipelines, not precision. Vehicles merged in dense queues are the misses, and they cause most of the ID switches. Check that before feeding video trajectories to a safety tool.

Caveat: rendered video, not real footage. Real footage needs a learned detector; the YOLO adapter is in the repo but not tested.

Repo: [link]

If you've fused video and loop counts on a real project, how long a calibration window did you need?

#DroneVideo #ComputerVision #TrafficData #MultiObjectTracking #TransportationEngineering #NGSIM #Python
