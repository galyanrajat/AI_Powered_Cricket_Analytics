from pathlib import Path
import logging
import yaml
from modules import (
    video_downloader,
    video_processor,
    pose_estimation,
    bat_detection,
    metrics,
    overlay,
    evaluation,
    phase_segmentation,
    contact_detection
)

# ------------------ Setup Logger ------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------ Load Settings -----------------
SETTINGS_PATH = Path("config/settings.yaml")  # updated path

if not SETTINGS_PATH.exists():
    logger.error(f"Settings file not found: {SETTINGS_PATH}")
    raise FileNotFoundError(f"Settings file not found: {SETTINGS_PATH}")

with open(SETTINGS_PATH, "r") as f:
    config = yaml.safe_load(f)

VIDEO_URL = config["video_url"]
TARGET_FPS = config.get("fps", 30)
TARGET_RES = tuple(config.get("resolution", [1280, 720]))
OUTPUT_DIR = Path(config.get("output_dir", "output/"))
LOGS_DIR = Path(config.get("logs_dir", "logs/"))
MODELS_DIR = Path(config.get("models_dir", "models/"))

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR = Path("input")
INPUT_DIR.mkdir(exist_ok=True)

# ------------------ Main Pipeline -----------------
def main():
    logger.info("Starting AthleteRise Cover Drive Analysis Pipeline...")

    # STEP 1: Download video-only if not present
    input_video_path = INPUT_DIR / "video.mp4"
    try:
        if not input_video_path.exists():
            logger.info("Downloading video (video-only)...")
            video_downloader.run(
                url=VIDEO_URL,
                output_path=input_video_path  # Path object
            )
        else:
            logger.info("Video already exists, skipping download")
    except Exception as e:
        logger.error(f"Video download failed: {e}")
        return

    # STEP 2: Normalize video
    try:
        norm_video_path = Path(
            video_processor.run(
                input_video_path,
                target_fps=TARGET_FPS,
                target_res=TARGET_RES
            )
        )
        if not norm_video_path.exists():
            raise FileNotFoundError(f"Normalized video not found at {norm_video_path}")
    except Exception as e:
        logger.error(f"Video normalization failed: {e}")
        return

    # STEP 3: Pose estimation
    try:
        keypoints_file = Path(pose_estimation.run(norm_video_path))
        if not keypoints_file.exists():
            raise FileNotFoundError(f"Pose keypoints file not found at {keypoints_file}")
    except Exception as e:
        logger.error(f"Pose estimation failed: {e}")
        return

    # STEP 4: Bat detection
    try:
        bat_file = Path(bat_detection.run(norm_video_path))
        if not bat_file.exists():
            raise FileNotFoundError(f"Bat positions file not found at {bat_file}")
    except Exception as e:
        logger.error(f"Bat detection failed: {e}")
        return

    # STEP 5: Metrics computation
    try:
        metrics_file = Path(metrics.run(keypoints_file, bat_file))
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found at {metrics_file}")
    except Exception as e:
        logger.error(f"Metrics computation failed: {e}")
        return

    # STEP 6: Phase segmentation
    try:
        phases_file = Path(phase_segmentation.run(keypoints_file))
        if not phases_file.exists():
            logger.warning("Phase segmentation file not found, continuing without it")
            phases_file = None
    except Exception as e:
        logger.error(f"Phase segmentation failed: {e}")
        phases_file = None

    # STEP 7: Contact detection
    try:
        contact_file = Path(contact_detection.run(
            keypoints_file,
            bat_file,
            phases_file
        ))
        if not contact_file.exists():
            logger.warning("Contact detection file not found, continuing without it")
            contact_file = None
    except Exception as e:
        logger.error(f"Contact detection failed: {e}")
        contact_file = None

    # STEP 8: Overlay
    try:
        annotated_video = Path(overlay.run(
            norm_video_path,
            keypoints_file,
            bat_file,
            metrics_file,
            phases_csv=phases_file,
            contact_file=contact_file
        ))

        if not annotated_video.exists():
            logger.warning("Annotated video not found")
            annotated_video = None
    except Exception as e:
        logger.error(f"Overlay generation failed: {e}")
        annotated_video = None

    # STEP 9: Final evaluation
    try:
        eval_file = Path(evaluation.run(metrics_file, phases_file, contact_file))
        if not eval_file.exists():
            logger.warning("Evaluation JSON not found")
            eval_file = None
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        eval_file = None

    logger.info("Pipeline completed!")
    logger.info(f"Annotated video: {annotated_video or 'Not generated'}")
    logger.info(f"Evaluation JSON: {eval_file or 'Not generated'}")


if __name__ == "__main__":
    main()
