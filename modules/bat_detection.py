import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from modules.logger import get_logger
from modules.cache_manager import needs_update, update_cache
import urllib.request
from tqdm import tqdm

logger = get_logger("bat_detection")

VIDEO_PATH = Path("output/normalized_video.mp4")
YOLO_WEIGHTS = Path("models/yolov8n_bat.pt")
OUT_FILE = Path("output/bat_positions.csv")


def ensure_model(weights_path: Path):
    """Download YOLOv8 nano model if not found."""
    if not weights_path.exists():
        logger.warning(f"⚠ YOLO weights not found at {weights_path}")
        logger.info("⬇ Downloading YOLOv8 nano model as placeholder...")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        urllib.request.urlretrieve(url, weights_path)
        logger.info(f"Model downloaded to {weights_path}")


def run(video_path: Path = VIDEO_PATH, yolo_weights: Path = YOLO_WEIGHTS) -> Path:
    """
    Detect bats in each frame using YOLOv8.
    Output format: frame, x1, y1, x2, y2, confidence
    Supports multiple detections per frame.
    """
    video_path = Path(video_path)
    yolo_weights = Path(yolo_weights)

    # Ensure model weights exist
    ensure_model(yolo_weights)

    step_config = {"weights": str(yolo_weights)}

    # Skip if no update needed
    if OUT_FILE.exists() and not needs_update("bat_detection", [video_path, yolo_weights], step_config):
        logger.info("Bat detection already up-to-date — skipping.")
        return OUT_FILE

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    logger.info(f"Starting bat detection with YOLOv8 model: {yolo_weights}")
    model = YOLO(str(yolo_weights))

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = []

    for frame_idx in tqdm(range(total_frames), desc="Bat Detection"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3)  # YOLOv8 inference
        for r in results:
            if hasattr(r, "boxes") and r.boxes is not None:
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

    cap.release()

    # Save detections
    if detections:
        df = pd.DataFrame(detections)
    else:
        df = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "confidence"])

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    logger.info(f"Bat positions saved to {OUT_FILE} ({len(df)} detections)")

    # Update cache
    update_cache("bat_detection", [video_path, yolo_weights, OUT_FILE], step_config)
    return OUT_FILE


if __name__ == "__main__":
    run()
