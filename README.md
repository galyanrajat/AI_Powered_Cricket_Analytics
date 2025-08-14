# AtheleteRise project folder structure



athleterise_coverdrive/
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
│
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
