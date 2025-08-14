from pathlib import Path
import pandas as pd
import numpy as np
from modules.logger import get_logger
from modules.cache_manager import needs_update, update_cache

logger = get_logger("contact_detection")

IN_KPTS = Path("output/pose_keypoints.csv")
IN_BATS = Path("output/bat_positions.csv")
IN_PHASES = Path("output/phases.csv")
OUT_CONTACT = Path("output/contact.json")

def _vel(df: pd.DataFrame, k: int) -> np.ndarray:
    x = df[f"x_{k}"].values
    y = df[f"y_{k}"].values
    sh_w = np.maximum(1e-6, np.sqrt((df["x_11"]-df["x_12"])**2 + (df["y_11"]-df["y_12"])**2).values)
    vx = np.gradient(x)
    vy = np.gradient(y)
    return np.nan_to_num(np.sqrt(vx**2 + vy**2) / sh_w)

def run(pose_csv: Path = IN_KPTS, bat_csv: Path = IN_BATS, phases_csv: Path = IN_PHASES) -> Path:
    """Pick likely contact frame via combined wrist speed peak + bat-box speed peak near Downswing/Impact."""
    pose_csv, bat_csv, phases_csv = Path(pose_csv), Path(bat_csv), Path(phases_csv)
    step_config = {"pose": str(pose_csv), "bat": str(bat_csv), "phases": str(phases_csv)}

    if OUT_CONTACT.exists() and not needs_update("contact_detection", [pose_csv, bat_csv, phases_csv], step_config):
        logger.info("Contact already computed — skipping.")
        return OUT_CONTACT

    if not pose_csv.exists():
        raise FileNotFoundError(f"Pose keypoints missing: {pose_csv}")
    if not phases_csv.exists():
        raise FileNotFoundError(f"Phases file missing: {phases_csv}")

    kdf = pd.read_csv(pose_csv)
    wrist_v = np.maximum(_vel(kdf, 16), _vel(kdf, 15))  # right/left wrist

    # Bat speed proxy (bbox center speed)
    bdf = pd.read_csv(bat_csv) if bat_csv.exists() else pd.DataFrame(columns=["frame","x1","y1","x2","y2","confidence"])
    max_frame = int(kdf["frame"].max())
    cx = np.zeros(max_frame+1)
    cy = np.zeros(max_frame+1)
    if not bdf.empty:
        btop = (bdf.sort_values(["frame","confidence"], ascending=[True, False])
                   .drop_duplicates(subset=["frame"]))
        for _, r in btop.iterrows():
            f = int(r["frame"])
            cx[f] = (r["x1"] + r["x2"]) / 2.0
            cy[f] = (r["y1"] + r["y2"]) / 2.0
    cb = np.sqrt(np.gradient(cx)**2 + np.gradient(cy)**2)

    # normalize both using np.ptp()
    wrist_z = (wrist_v - wrist_v.min()) / (np.ptp(wrist_v) + 1e-6)
    bat_z   = (cb - cb.min()) / (np.ptp(cb) + 1e-6)
    score_curve = 0.6 * wrist_z + 0.4 * bat_z

    # Limit search window around downswing/impact phases if present
    pdf = pd.read_csv(phases_csv)
    window = (0, max_frame)
    for _, r in pdf.iterrows():
        if r["phase"] in ["Downswing", "Impact"]:
            window = (int(r["start"]), int(r["end"]))
            break

    s, e = window
    cand_idx = np.argmax(score_curve[s:e+1]) + s
    out = {"contact_frame": int(cand_idx), "window_start": int(s), "window_end": int(e)}

    import json
    with open(OUT_CONTACT, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Contact frame estimated: {out['contact_frame']} (search window {s}-{e}) → {OUT_CONTACT}")
    update_cache("contact_detection", [pose_csv, bat_csv, phases_csv, OUT_CONTACT], step_config)
    return OUT_CONTACT

