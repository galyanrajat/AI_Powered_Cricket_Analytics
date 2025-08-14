# AtheleteRise project folder structure

# Objective
Build a Python-based system that processes the entire cricket video in real time (no screenshots/keyframe exports), performs pose estimation frame-by-frame, and outputs an annotated video with live overlays and a final shot evaluation.

# project Structure

athleterise/
│
├── main.py                               # Orchestrates full pipeline
│
├── modules/
│   ├── video_downloader.py               # Download video from YouTube
│   ├── video_processor.py                # Normalize FPS/resolution
│   ├── pose_estimation.py                 # MediaPipe Pose keypoints
│   ├── bat_detection.py                   # YOLOv8 bat detection & tracking
│   ├── metrics.py                         # Calculate biomechanical metrics
│   ├── overlay.py                         # Draw skeleton, bat boxes, metrics
│   ├── evaluation.py                      # Final scoring & feedback
│   ├── logger.py                          # Unified logging
│   ├── cache_manager.py                   # Step caching system
│   |__ contact_detection.py               # moment of bat-ball contact
|   |__ phase_sentimentation.py            # Divide the cover drive into distinct phases
├── models/
│   ├── yolov8n_bat.pt                     # YOLO model weights for bat detection
│
├── config/
│   ├── settings.yaml                      # Thresholds, model paths, configs
│
├── output/
│   ├── annotated_video.mp4
│   ├── evaluation.json
│   ├── metrics_log.csv
│   ├── elbow_spine_chart.png
│
├── logs/
│   ├── pipeline.log
│
├── requirements.txt
└── README.md

# Deliverables
● cover_drive_analysis_realtime.py (main script)
● /output/
○ annotated_video.mp4 (with overlays, full-length)
○ evaluation.json (or .txt) with category scores & comments
● requirements.txt (or environment.yml)
● README.md

#  Technologies & Tools
Programming: Python 3.x

Libraries: OpenCV, MediaPipe / OpenPose, Ultralytics YOLOv8, Pandas, NumPy, tqdm

Supporting Tools: JSON/CSV for structured outputs, logging for pipeline monitoring

# Real-Time Video Processing

Processes full video frames sequentially using OpenCV.

Normalizes FPS/resolution while maintaining near real-time performance.

# Pose Estimation

Extracts keypoints for head, shoulders, elbows, wrists, hips, knees, and ankles.

Computes angles and alignment metrics for biomechanical assessment.

# Bat Detection

Detects bat position using YOLOv8.

Automatically downloads weights if missing, ensuring seamless setup.

Outputs frame-wise bat coordinates for analysis.

# Contact Detection

Identifies likely bat-ball contact frames using wrist velocity and motion heuristics.

Critical for assessing impact and swing timing.

# Phase Segmentation

Divides the shot into six phases: Stance → Stride → Downswing → Impact → Follow-through → Recovery.

Enables phase-wise scoring and targeted feedback.

# Metrics & Scoring

Evaluates footwork, head alignment, swing control, balance, and follow-through.

Provides final summary scores (1–10) and actionable feedback per category.

Supports trend analysis using frame-to-frame metric variations.

# Outputs

Annotated video with skeleton overlays, bat positions, phase indicators, and real-time metrics.

CSV/JSON outputs with detailed frame-wise detection and phase data.

Optional HTML/PDF report summarizing performance metrics.

# Pipeline Optimization & Caching

Uses caching to skip re-processing unchanged videos.

Optimized for ≥10 FPS performance on CPU for real-time analysis.
