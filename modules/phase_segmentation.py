from pathlib import Path
import pandas as pd
import numpy as np
from modules.logger import get_logger
from modules.cache_manager import needs_update, update_cache

logger = get_logger("phase_segmentation")

IN_KPTS = Path("output/pose_keypoints.csv")
OUT_PHASES = Path("output/phases.csv")

# Helper: finite difference speed (per-frame Euclidean velocity) for a keypoint
def _vel(df: pd.DataFrame, k: int) -> np.ndarray:
    x = df[f"x_{k}"].values
    y = df[f"y_{k}"].values
    # normalize by shoulder width to reduce scale sensitivity
    sh_w = np.maximum(1e-6, np.sqrt((df["x_11"]-df["x_12"])**2 + (df["y_11"]-df["y_12"])**2).values)
    vx = np.gradient(x)
    vy = np.gradient(y)
    v = np.sqrt(vx**2 + vy**2) / sh_w
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

# Simple shoulder-line angle (torso orientation proxy)
def _torso_angle(df: pd.DataFrame) -> np.ndarray:
    dx = (df["x_12"] - df["x_11"]).values
    dy = (df["y_12"] - df["y_11"]).values
    ang = np.degrees(np.arctan2(dy, dx))
    return np.nan_to_num(ang)

def run(pose_csv: Path = IN_KPTS) -> Path:
    """Heuristic phase segmentation using wrist velocity + torso angle dynamics."""
    pose_csv = Path(pose_csv)
    step_config = {"pose_csv": str(pose_csv)}

    if OUT_PHASES.exists() and not needs_update("phase_segmentation", [pose_csv], step_config):
        logger.info("Phases already computed — skipping.")
        return OUT_PHASES

    if not pose_csv.exists():
        raise FileNotFoundError(f"Pose keypoints missing: {pose_csv}")

    df = pd.read_csv(pose_csv)
    # assume right-handed: right wrist(16), right elbow(14); fall back to left if right missing
    rw_v = _vel(df, 16)
    lw_v = _vel(df, 15)
    wrist_v = np.maximum(rw_v, lw_v)  # handle handedness automatically

    torso = _torso_angle(df)
    d_torso = np.abs(np.gradient(torso))

    # Heuristics (tune thresholds if needed)
    idle_v = np.percentile(wrist_v, 20)
    swing_v = np.percentile(wrist_v, 75)
    rotate_thr = np.percentile(d_torso, 70)

    # State machine
    frames = df["frame"].values
    phases = []
    state = "Stance"
    start_idx = int(frames[0])

    def push_phase(name, s, e):
        phases.append({"phase": name, "start": int(s), "end": int(e)})

    for i, f in enumerate(frames[1:], start=1):
        v = wrist_v[i]
        rot = d_torso[i]

        if state == "Stance":
            if v > idle_v * 1.5 or rot > rotate_thr * 0.7:
                push_phase("Stance", start_idx, frames[i-1])
                state = "Stride"
                start_idx = f

        elif state == "Stride":
            # approaching bat lift / early motion
            if v > swing_v * 0.7 or rot > rotate_thr:
                push_phase("Stride", start_idx, frames[i-1])
                state = "Downswing"
                start_idx = f

        elif state == "Downswing":
            # peak wrist velocity typically near impact
            # drop after peak means impact happened
            if i > 2 and wrist_v[i-1] > wrist_v[i] and wrist_v[i-1] > swing_v:
                push_phase("Downswing", start_idx, frames[i-1])
                state = "Impact"
                start_idx = f

        elif state == "Impact":
            # Impact is brief; move quickly to Follow-through
            if v < swing_v * 0.9:
                push_phase("Impact", start_idx, frames[i-1])
                state = "Follow-through"
                start_idx = f

        elif state == "Follow-through":
            # velocities decaying
            if v < idle_v * 1.2 and rot < rotate_thr * 0.5:
                push_phase("Follow-through", start_idx, frames[i-1])
                state = "Recovery"
                start_idx = f

        elif state == "Recovery":
            # end of action
            pass

    # flush last state
    push_phase(state, start_idx, int(frames[-1]))

    pd.DataFrame(phases).to_csv(OUT_PHASES, index=False)
    logger.info(f"Phases saved: {OUT_PHASES}")
    update_cache("phase_segmentation", [pose_csv, OUT_PHASES], step_config)
    return OUT_PHASES
