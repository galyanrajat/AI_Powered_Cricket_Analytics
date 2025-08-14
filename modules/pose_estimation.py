import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from modules.logger import get_logger
from modules.cache_manager import needs_update, update_cache

logger = get_logger("pose_estimation")

PROCESSED_VIDEO = Path("output/normalized_video.mp4")
KEYPOINTS_FILE = Path("output/pose_keypoints.csv")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def run(video_path: Path = PROCESSED_VIDEO) -> Path:
    """
    Runs MediaPipe Pose on the normalized video and saves keypoints to a CSV file.
    Each row contains frame number + (x,y,visibility) for each landmark.
    """
    step_config = {"pose_model": "mediapipe", "landmarks": 33}

    if KEYPOINTS_FILE.exists() and not needs_update("pose_estimation", [video_path], step_config):
        logger.info("Pose keypoints already up-to-date — skipping.")
        return KEYPOINTS_FILE

    if not video_path.exists():
        raise FileNotFoundError(f"Normalized video not found: {video_path}")

    logger.info("Starting pose estimation using MediaPipe Pose...")

    cap = cv2.VideoCapture(str(video_path))
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

    all_keypoints = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            row = [frame_idx]
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.visibility])
            all_keypoints.append(row)
        else:
            # Fill with NaNs if no detection
            row = [frame_idx] + [None] * (33 * 3)
            all_keypoints.append(row)

        frame_idx += 1
        if frame_idx % 50 == 0:
            logger.info(f"Processed {frame_idx} frames for pose...")

    cap.release()
    pose.close()

    # Prepare column names
    columns = ["frame"]
    for i in range(33):
        columns.extend([f"x_{i}", f"y_{i}", f"v_{i}"])

    df = pd.DataFrame(all_keypoints, columns=columns)
    KEYPOINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(KEYPOINTS_FILE, index=False)

    logger.info(f"Pose keypoints saved to {KEYPOINTS_FILE}")

    update_cache("pose_estimation", [video_path, KEYPOINTS_FILE], step_config)
    return KEYPOINTS_FILE
