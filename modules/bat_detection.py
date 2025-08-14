import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from modules.logger import get_logger
from modules.cache_manager import needs_update, update_cache

logger = get_logger("bat_detection")

VIDEO_PATH = Path("output/normalized_video.mp4")
YOLO_WEIGHTS = Path("models/yolov8n_bat.pt")
OUT_FILE = Path("output/bat_positions.csv")

def run(video_path: Path = VIDEO_PATH, yolo_weights: Path = YOLO_WEIGHTS) -> Path:
    """
    Detect bats in each frame using YOLOv8.
    Output format: frame, x1, y1, x2, y2, confidence
    Supports multiple detections per frame.
    """
    # Ensure both inputs are Path objects
    video_path = Path(video_path)
    yolo_weights = Path(yolo_weights)

    step_config = {"weights": str(yolo_weights)}

    # Skip if no update needed
    if OUT_FILE.exists() and not needs_update("bat_detection", [video_path, yolo_weights], step_config):
        logger.info("Bat detection already up-to-date — skipping.")
        return OUT_FILE

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not yolo_weights.exists():
        logger.warning(f"⚠ YOLO weights not found at {yolo_weights}, bat detection will be empty.")
        pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "confidence"]).to_csv(OUT_FILE, index=False)
        return OUT_FILE

    logger.info(f"🏏 Starting bat detection with YOLOv8 model: {yolo_weights}")
    model = YOLO(str(yolo_weights))

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.3, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({
                    "frame": frame_idx,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf
                })

        frame_idx += 1
        if frame_idx % 50 == 0:
            logger.info(f"Processed {frame_idx} frames for bat detection...")

    cap.release()

    # Save detections
    df = pd.DataFrame(detections)
    df.to_csv(OUT_FILE, index=False)
    logger.info(f"Bat positions saved to {OUT_FILE} ({len(df)} detections)")

    update_cache("bat_detection", [video_path, yolo_weights, OUT_FILE], step_config)
    return OUT_FILE
