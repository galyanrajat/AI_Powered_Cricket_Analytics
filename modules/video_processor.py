import cv2
from pathlib import Path
from modules.logger import get_logger
from modules.cache_manager import needs_update, update_cache

logger = get_logger("video_processor")

def run(video_path: Path, target_fps: int, target_res: tuple) -> Path:
    """
    Normalize video FPS and resolution.
    Returns path to normalized video.
    """
    output_path = Path("output/normalized_video.mp4")
    step_config = {
        "target_fps": target_fps,
        "target_res": target_res
    }

    # Skip if up-to-date
    if output_path.exists() and not needs_update("video_processor", [output_path], step_config):
        logger.info("Normalized video already exists — skipping.")
        return output_path

    logger.info(f"Normalizing video to {target_fps} FPS, resolution {target_res}")

    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, target_fps, target_res)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, target_res)
        out.write(frame_resized)

    cap.release()
    out.release()

    logger.info(f"Normalized video saved at {output_path}")
    update_cache("video_processor", [output_path], step_config)
    return output_path
