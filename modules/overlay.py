import cv2
import pandas as pd
from pathlib import Path
from modules.logger import get_logger
from modules.cache_manager import needs_update, update_cache

logger = get_logger("overlay")

VIDEO_PATH = Path("output/normalized_video.mp4")
KEYPOINTS_FILE = Path("output/pose_keypoints.csv")
BAT_FILE = Path("output/bat_positions.csv")
METRICS_FILE = Path("output/metrics_log.csv")
OUT_VIDEO = Path("output/annotated_video.mp4")

POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (23, 25), (25, 27), (24, 26),
    (26, 28), (23, 24), (11, 23), (12, 24)
]

def _draw_pose(frame, row, w, h):
    for i in range(33):
        x = row.get(f"x_{i}"); y = row.get(f"y_{i}"); v = row.get(f"v_{i}")
        if pd.notnull(x) and pd.notnull(y) and (v is None or v > 0.3):
            cv2.circle(frame, (int(x*w), int(y*h)), 3, (255, 255, 255), -1)
    for a, b in POSE_CONNECTIONS:
        xa, ya, va = row.get(f"x_{a}"), row.get(f"y_{a}"), row.get(f"v_{a}")
        xb, yb, vb = row.get(f"x_{b}"), row.get(f"y_{b}"), row.get(f"v_{b}")
        if pd.notnull(xa) and pd.notnull(ya) and pd.notnull(xb) and pd.notnull(yb):
            if (va is None or va>0.3) and (vb is None or vb>0.3):
                cv2.line(frame, (int(xa*w), int(ya*h)), (int(xb*w), int(yb*h)), (255,255,255), 2)

def _draw_bat(frame, bat_rows):
    if not bat_rows: return
    for b in bat_rows:
        x1, y1, x2, y2, conf = b
        if pd.notnull(x1) and pd.notnull(y1) and pd.notnull(x2) and pd.notnull(y2):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (60, 220, 60), 2)
            cv2.putText(frame, f"Bat {conf:.2f}", (int(x1), max(15,int(y1-8))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60,220,60), 2)

def _draw_metrics_panel(frame, mrow, cfg):
    cv2.rectangle(frame, (10,10), (330,110), (0,0,0), -1)
    cv2.rectangle(frame, (10,10), (330,110), (255,255,255), 1)
    elbow = mrow.get("elbow_angle"); spine = mrow.get("spine_angle")
    hk = mrow.get("head_knee_distance"); foot = mrow.get("foot_angle")
    elbow_thr = cfg.get("elbow_angle_threshold", 110)
    spine_thr = cfg.get("spine_lean_threshold", 10)
    hk_thr = cfg.get("head_knee_distance_threshold", 15)
    def flag(val, thr, good_when_less=True):
        if pd.isnull(val): return "NULL"
        ok = val < thr if good_when_less else val > thr
        return "Done" if ok else "Note done"
    lines = [
        f"Elbow Angle: {None if pd.isnull(elbow) else round(elbow,1)}°  {flag(elbow, elbow_thr, good_when_less=False)}",
        f"Spine Lean : {None if pd.isnull(spine) else round(spine,1)}°  {flag(spine, spine_thr)}",
        f"Head-Knee  : {None if pd.isnull(hk) else round(hk,3)}  {flag(hk, hk_thr)}",
        f"Foot Angle : {None if pd.isnull(foot) else round(foot,1)}°"
    ]
    y = 35
    for text in lines:
        cv2.putText(frame, text, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255),1)
        y+=20

def run(
    video_path: Path = VIDEO_PATH,
    keypoints_csv: Path = KEYPOINTS_FILE,
    bat_csv: Path = BAT_FILE,
    metrics_csv: Path = METRICS_FILE,
    phases_csv: Path = None,
    contact_file: Path = None,
    config: dict = None
) -> Path:
    cfg = config or {}
    step_config = {"thresholds": cfg}

    if OUT_VIDEO.exists() and not needs_update("overlay",
        [video_path, keypoints_csv, bat_csv, metrics_csv, phases_csv, contact_file],
        step_config
    ):
        logger.info("Annotated video up-to-date — skipping.")
        return OUT_VIDEO

    if not video_path.exists(): raise FileNotFoundError(f"Video missing: {video_path}")
    if not keypoints_csv.exists(): raise FileNotFoundError(f"Keypoints missing: {keypoints_csv}")
    if not metrics_csv.exists(): raise FileNotFoundError(f"Metrics missing: {metrics_csv}")

    kp_df = pd.read_csv(keypoints_csv)
    bat_df = pd.read_csv(bat_csv) if bat_csv and bat_csv.exists() else pd.DataFrame(columns=["frame"])
    met_df = pd.read_csv(metrics_csv)

    # Phase map
    phase_map = {}
    if phases_csv and phases_csv.exists():
        pdf = pd.read_csv(phases_csv)
        for _, r in pdf.iterrows():
            for f in range(int(r["start"]), int(r["end"])+1):
                phase_map[f] = r["phase"]

    # Contact frame
    contact_frame = None
    if contact_file and contact_file.exists():
        import json
        with open(contact_file) as f:
            contact_frame = json.load(f).get("contact_frame")

    # Bat map
    bat_map = {}
    if not bat_df.empty:
        for frame_num, group in bat_df.groupby("frame"):
            bat_map[frame_num] = [(row["x1"], row["y1"], row["x2"], row["y2"], row["confidence"]) for _, row in group.iterrows()]

    met_map = met_df.set_index("frame").to_dict("index")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    frame_idx = 0
    logger.info("Drawing overlays...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        if not kp_df[kp_df["frame"]==frame_idx].empty:
            _draw_pose(frame, kp_df[kp_df["frame"]==frame_idx].iloc[0], w, h)
        if frame_idx in bat_map: _draw_bat(frame, bat_map[frame_idx])
        if frame_idx in met_map: _draw_metrics_panel(frame, met_map[frame_idx], cfg)

        # Phase display
        if frame_idx in phase_map:
            cv2.putText(frame, f"Phase: {phase_map[frame_idx]}", (10, h-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)

        # Contact frame highlight
        if contact_frame is not None and frame_idx==contact_frame:
            cv2.putText(frame, "CONTACT", (w//2-80, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        cv2.putText(frame, "AthleteRise: Cover Drive Analysis", (10,h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240),2)
        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            logger.info(f"Overlayed {frame_idx} frames...")

    cap.release()
    out.release()
    logger.info(f"Annotated video saved: {OUT_VIDEO}")

    update_cache("overlay", [video_path, keypoints_csv, bat_csv, metrics_csv, OUT_VIDEO], step_config)
    return OUT_VIDEO
