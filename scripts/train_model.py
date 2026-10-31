"""Train and save the predictor from the locally synced player-match dataset."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engines import FEATURES, calibrate_consistency_k, prepare_match_scores, train_predictor

DATA = ROOT / "data" / "processed" / "player_match_stats_api.csv"
MODEL_DIR = ROOT / "models"


def main():
    if not DATA.exists():
        raise SystemExit("Run scripts/sync_pitchapi.py first to build the API dataset.")
    raw = pd.read_csv(DATA)
    scored = prepare_match_scores(raw)
    k = calibrate_consistency_k(scored)
    model, training_rows, metrics = train_predictor(scored, k=k)
    if model is None:
        raise SystemExit(metrics.get("error", "Model training could not proceed."))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "features": FEATURES,
        "consistency_k": k,
        "data_source": "PitchAPI",
        "dataset": str(DATA.relative_to(ROOT)),
        "last_match": str(scored["match_date"].max()),
        "training_rows": len(training_rows),
    }
    joblib.dump(artifact, MODEL_DIR / "player_form_predictor.joblib", compress=3)
    report = {**metrics, "consistency_k": k, "dataset_rows": len(raw), "players": int(raw.player_id.nunique()),
              "matches": int(raw.match_id.nunique()), "latest_match": str(scored["match_date"].max())}
    (MODEL_DIR / "player_form_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved model to {MODEL_DIR / 'player_form_predictor.joblib'}")


if __name__ == "__main__":
    main()
