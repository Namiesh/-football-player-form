# Football Player Form Lab

Streamlit app for Premier League player form, next-appearance performance and team xPoints. Football data is read from the provided Opta S3 bucket; the app and training pipeline make no calls to a football-data API.

## S3 dataset

The importer reads only `matchstats.json` under `scrape-opta-data/football/<year>/premier-league-england/` and `scrape-opta-data/continuous-scrape-data/football/<year>/premier-league-england/`. It caches objects by S3 ETag and writes:

- `data/processed/player_match_stats_s3.csv` — historical and current-season player appearances.
- `data/processed/player_match_stats_2026_2027.csv` — current season.

Configure rotated, read-only AWS credentials locally with the AWS CLI. Do not put keys in this repository or send them in chat. Example:

```powershell
aws configure --profile football-form-readonly
$env:AWS_PROFILE = "football-form-readonly"
$env:AWS_REGION = "ap-south-1"
python scripts/sync_s3_opta.py
```

The importer defaults to calendar years 2023–2027; each match's season is read from its Opta calendar. The last sync found 1,220 matches across 2023/24–2026/27, including 80 matches in 2026/27 through 25 October 2026. The continuous bucket should be synced after it receives new matchstats files.

## Train and run

```powershell
python scripts/train_model.py
streamlit run app.py
```

The trainer saves `models/player_form_predictor.joblib` and `models/player_form_metrics.json`. It evaluates a chronological holdout against a recent-form baseline; inspect both MAEs before relying on predictions. The target is the app's formula-derived match-score percentile, not an independently observed ground-truth rating.

On the current snapshot, the model MAE is 7.66 form points versus 8.21 for the recent-form baseline, with R² 0.055. This is a small improvement with weak explanatory power; predictions are experimental, not a claim of high accuracy. Data currently ends 25 October 2026, so sync again when newer matches arrive.

## Method and data limits

Match scores use role-relative percentiles from available per-90 statistics. Form combines recency-weighted F5/F10/F15 with minutes confidence. The ML engine uses strictly prior appearances, including `days_rest`, and reports a chronological holdout.

The sampled S3 matchstats feed has stable player IDs, minutes, goals, assists, shots, passing, tackles, clearances and cards. It does not include shot-level xG/xA, explicit injuries, interceptions or recoveries. The app reduces rating confidence after a long appearance gap, but cannot determine whether a gap was caused by injury, rotation, suspension, transfer or missing data. Add an explicit injury-history feed to model injury status reliably.

The linked [xDanger article](https://marclamberts.medium.com/introducing-xdanger-one-ecosystem-for-shot-models-possession-value-models-and-ratings-a00f51663a1d) describes shot and possession models trained from event-level context. That feed is not present in the sampled S3 paths, so this app does not claim to reproduce xDanger. Team xPoints uses independent Poisson score probabilities and `3 × P(win) + P(draw)`, with recent goals for/against as a clearly labeled rate proxy because the supplied files have no xG.
