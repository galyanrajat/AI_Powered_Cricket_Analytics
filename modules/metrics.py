import math
import pandas as pd
from pathlib import Path
from modules.logger import get_logger
from modules.cache_manager import needs_update, update_cache

logger = get_logger("metrics")

POSE_FILE = Path("output/pose_keypoints.csv")
BAT_FILE = Path("output/bat_positions.csv")
METRICS_FILE = Path("output/metrics_log.csv")

# Helper functions
def angle_3pts(a, b, c):
    """Return angle at point b (in degrees) given three points (x,y)."""
    if None in a or None in b or None in c:
        return None
    try:
        ang = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0]) -
            math.atan2(a[1] - b[1], a[0] - b[0])
        )
        return abs(ang) if ang >= 0 else abs(ang + 360)
    except:
        return None

def distance(a, b):
    """Euclidean distance between two points."""
    if None in a or None in b:
        return None
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def run(pose_file: Path = POSE_FILE, bat_file: Path = BAT_FILE, config: dict = {}) -> Path:
    """
    Computes biomechanical metrics for each frame:
    - Front elbow angle (shoulder–elbow–wrist)
    - Spine lean (hip–shoulder vs vertical)
    - Head-over-knee distance
    - Front foot direction (toe vs horizontal axis)
    """
    step_config = {
        "thresholds": {
            "elbow": config.get("elbow_angle_threshold"),
            "spine": config.get("spine_lean_threshold"),
            "head_knee": config.get("head_knee_distance_threshold")
        }
    }

    if METRICS_FILE.exists() and not needs_update("metrics", [pose_file, bat_file], step_config):
        logger.info("Metrics already up-to-date — skipping.")
        return METRICS_FILE

    if not pose_file.exists():
        raise FileNotFoundError(f"Pose keypoints file missing: {pose_file}")

    # Load data
    pose_df = pd.read_csv(pose_file)
    bat_df = pd.read_csv(bat_file) if bat_file.exists() else pd.DataFrame(columns=["frame"])

    metrics = []

    for idx, row in pose_df.iterrows():
        frame = row["frame"]

        # Example: Assume front arm = left arm (Landmarks: 11=shoulder, 13=elbow, 15=wrist)
        shoulder = (row["x_11"], row["y_11"])
        elbow    = (row["x_13"], row["y_13"])
        wrist    = (row["x_15"], row["y_15"])
        elbow_angle = angle_3pts(shoulder, elbow, wrist)

        # Spine lean: Compare shoulder-hip line vs vertical axis
        hip = (row["x_23"], row["y_23"])
        dy = shoulder[1] - hip[1]
        dx = shoulder[0] - hip[0]
        spine_angle = abs(math.degrees(math.atan2(dx, dy)))

        # Head-over-knee distance (Landmarks: head=0, front knee=25)
        head = (row["x_0"], row["y_0"])
        knee = (row["x_25"], row["y_25"])
        head_knee_dist = distance(head, knee)

        # Foot direction (toe vs horizontal)
        toe = (row["x_31"], row["y_31"])
        heel = (row["x_29"], row["y_29"])
        foot_angle = abs(math.degrees(math.atan2(toe[1] - heel[1], toe[0] - heel[0])))

        # Get bat detection (if available)
        bat_row = bat_df[bat_df["frame"] == frame]
        if not bat_row.empty:
            bat_coords = bat_row.iloc[0][["x1", "y1", "x2", "y2"]].tolist()
        else:
            bat_coords = [None, None, None, None]

        metrics.append([
            frame, elbow_angle, spine_angle, head_knee_dist, foot_angle,
            *bat_coords
        ])

    metrics_df = pd.DataFrame(metrics, columns=[
        "frame", "elbow_angle", "spine_angle", "head_knee_distance", "foot_angle",
        "bat_x1", "bat_y1", "bat_x2", "bat_y2"
    ])
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_FILE, index=False)

    logger.info(f"Metrics saved to {METRICS_FILE}")

    update_cache("metrics", [pose_file, bat_file, METRICS_FILE], step_config)
    return METRICS_FILE
