from pathlib import Path
import json
import pandas as pd
import numpy as np
from modules.logger import get_logger
from modules.cache_manager import needs_update, update_cache

logger = get_logger("evaluation")

METRICS_FILE = Path("output/metrics_log.csv")
PHASES_FILE = Path("output/phases.csv")
CONTACT_FILE = Path("output/contact.json")
OUT_JSON = Path("output/evaluation.json")

def _safe_mean(series):
    s = series.dropna()
    return float(s.mean()) if len(s) else None

def _clip01(x): 
    return max(0.0, min(1.0, x))

def run(metrics_csv: Path = METRICS_FILE,
        phases_csv: Path = PHASES_FILE,
        contact_json: Path = CONTACT_FILE,
        config: dict = None) -> Path:
    """Compute 1–10 scores + feedback and save to evaluation.json"""
    metrics_csv, phases_csv, contact_json = Path(metrics_csv), Path(phases_csv), Path(contact_json)
    cfg = config or {}
    step_config = {"cfg": cfg}

    if OUT_JSON.exists() and not needs_update("evaluation", [metrics_csv, phases_csv, contact_json], step_config):
        logger.info("Evaluation already up-to-date — skipping.")
        return OUT_JSON

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics missing: {metrics_csv}")

    mdf = pd.read_csv(metrics_csv)

    # Pull representative values
    elbow = _safe_mean(mdf["elbow_angle"])                  # larger is generally “more extension”
    spine = _safe_mean(mdf["spine_angle"])                  # smaller lean is better (straighter)
    hk    = _safe_mean(mdf["head_knee_distance"])           # smaller distance (still, steady head) is better
    foot  = _safe_mean(mdf["foot_angle"])                   # moderate open stance

    # Convert to 0-1, then scale to 1-10. Basic linear maps with clamp.
    # Footwork: reward foot angle near ~30° (open but not too much)
    foot_target = 30.0
    foot_norm = 1.0 - min(1.0, abs((foot or foot_target) - foot_target) / 40.0)

    # Head Position: reward small head-knee distance
    hk_thr = cfg.get("head_knee_distance_threshold", 15)
    hk_norm = 1.0 - min(1.0, (hk or hk_thr) / (2*hk_thr))

    # Swing Control: penalize excessive elbow angle variability — approximate by elbow mean vs. target
    elbow_target = cfg.get("elbow_angle_threshold", 110)  # treat as desirable minimum
    elbow_norm = _clip01(((elbow or elbow_target) - elbow_target) / 40.0)  # more than threshold → good up to +40°

    # Balance: penalize spine lean
    spine_thr = cfg.get("spine_lean_threshold", 10)
    spine_norm = 1.0 - min(1.0, (spine or spine_thr) / (2*spine_thr))

    # Follow-through: encourage continuation (proxy: post-impact wrist speed decay smoothness)
    # If we have phases, longer follow-through than downswing is generally good.
    try:
        pdf = pd.read_csv(phases_csv)
        downswing = pdf[pdf["phase"] == "Downswing"].iloc[0] if (pdf["phase"] == "Downswing").any() else None
        follow = pdf[pdf["phase"] == "Follow-through"].iloc[0] if (pdf["phase"] == "Follow-through").any() else None
        if downswing is not None and follow is not None:
            ds_len = downswing["end"] - downswing["start"] + 1
            ft_len = follow["end"] - follow["start"] + 1
            ft_ratio = ft_len / max(1, ds_len)
            ft_norm = max(0.0, min(1.0, (ft_ratio - 0.5) / 1.0))  # >0.5x is okay; 1.5x+ is great
        else:
            ft_norm = 0.6
    except Exception:
        ft_norm = 0.6

    scores01 = {
        "Footwork": foot_norm,
        "Head Position": hk_norm,
        "Swing Control": elbow_norm,
        "Balance": spine_norm,
        "Follow-through": ft_norm
    }
    scores = {k: round(1 + v * 9, 1) for k, v in scores01.items()}

    # Feedback lines
    fb = {}
    fb["Footwork"] = "Nice base. Try adjusting your front-foot angle closer to ~30° for cleaner alignment." if foot_norm >= 0.6 \
        else "Work on planting the front foot ~30° open; over/under rotation is affecting stability."
    fb["Head Position"] = "Head stays steady over the knee — solid!" if hk_norm >= 0.6 \
        else "Keep your head more stable through the shot; minimize side drift."
    fb["Swing Control"] = "Elbow extension looks controlled through impact." if elbow_norm >= 0.6 \
        else "Avoid over-folding the elbow before impact; feel the bat extend through the line."
    fb["Balance"] = "Spine angle is compact — good balance." if spine_norm >= 0.6 \
        else "Reduce spine lean; think ‘tall chest’ to keep balance over the stance."
    fb["Follow-through"] = "Follow-through duration is healthy; let the bat finish high." if ft_norm >= 0.6 \
        else "Let the bat continue naturally after impact; avoid cutting the swing short."

    out = {
        "scores": scores,
        "feedback": fb
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Evaluation saved: {OUT_JSON}")
    update_cache("evaluation", [metrics_csv, phases_csv, contact_json, OUT_JSON], step_config)
    return OUT_JSON
