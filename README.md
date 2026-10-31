# Football Player Form Lab

A Streamlit app with two separate components:

1. **Player Form Rating Engine** — position-relative match scores, recency weighted form, consistency, minutes confidence and historical comparison.
2. **ML Prediction Engine** — a time-ordered Random Forest model trained to predict a player's next appearance score and label the expected form direction.

## Current and historical data

The current-data importer uses [PitchAPI](https://pitchapi.dev/), which documents match-level player stats, stable player IDs, 2026-era data, and a free plan without a published request allowance. Its fair-use burst guard still applies. The default import is the Premier League 2025/26 and 2026/27 seasons. The upstream match stats contain minutes, goals, assists, xG, shots, passing, tackles, interceptions, duels, clearances and other position-dependent values. The importer requests one player-stats response per finished match, waits between uncached calls and saves each response, so later runs only request newly finished matches.

Create a free API key, copy `.env.example` to `.env`, and set `PITCHAPI_API_KEY` there. Keep `.env` private. Then build or update `data/processed/player_match_stats_api.csv`:

```powershell
python scripts/sync_pitchapi.py
```

The sync also writes the 2026/27-only dataset to `data/processed/player_match_stats_2026_27.csv`. Change `PITCHAPI_LEAGUE` and `PITCHAPI_SEASONS` in `.env` to select other supported competitions/seasons. The Streamlit app prefers the PitchAPI dataset when present. The previous StatsBomb file remains as an archived fallback only.

Train and save the predictor and its chronological holdout metrics:

```powershell
python scripts/train_model.py
```

The trained artifact is written to `models/player_form_predictor.joblib` and its report to `models/player_form_metrics.json`.

Then launch the app:

```powershell
streamlit run app.py
```

## Method

Each match metric is converted to a percentile within position group and competition-season. Negative events are reverse scored. Metrics are combined into a 0–100 match score. Current form is the supplied recency-decay formula, with match weights also scaled by minutes played. The overall view is `0.50 × F5 + 0.30 × F10 + 0.20 × F15`. A minutes-confidence adjustment pulls low-minute estimates toward 50.

The ML target is the player's next-match performance score. Features are built strictly from earlier appearances and evaluated with a chronological holdout to reduce leakage. This is a forecasting baseline, not a claim that the formula score is an objective ground-truth rating. xG and expected assists are supplied by the provider. Opponent strength and team form are not included in this first version.

## Dataset notes

The initial Premier League sync uses one call for the league list, one per season for fixtures, one per uncached finished match for its player-stat table, and a lineup call only when positions are missing. Player IDs are retained as the provider's stable identifiers. You can extend the same importer to other leagues exposed by the API.
