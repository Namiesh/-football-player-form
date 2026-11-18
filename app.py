from pathlib import Path

import pandas as pd
import streamlit as st

from engines import calibrate_consistency_k, current_summary, latest_feature_row, prepare_match_scores, project_next_form, project_team_xpoints, team_match_xpoints, train_predictor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "player_match_stats_s3.csv"

st.set_page_config(page_title="Player Form Lab", page_icon="⚽", layout="wide")
st.title("⚽ Football Player Form Lab")
st.caption("A historical form rating engine and a separate next-performance ML predictor.")


@st.cache_data(show_spinner="Preparing match scores…")
def load_data(path: str, modified: float):
    raw = pd.read_csv(path)
    return prepare_match_scores(raw)


@st.cache_resource(show_spinner="Training the chronological ML baseline…")
def fit_model(path: str, modified: float, k: float):
    return train_predictor(load_data(path, modified), k=k)


if not DATA.exists():
    st.warning("The historical player-match dataset has not been built yet.")
    st.code("python scripts/sync_s3_opta.py", language="powershell")
    st.markdown("Configure a read-only AWS profile locally. This importer reads only the provided S3 matchstats files and makes no football-data API calls.")
    st.stop()

df = load_data(str(DATA), DATA.stat().st_mtime)
if df.empty:
    st.error("The dataset is empty. Rebuild it from S3 with `python scripts/sync_s3_opta.py`.")
    st.stop()
latest_data_date = df["match_date"].max()
xpoints_matches = team_match_xpoints(df)
if pd.notna(latest_data_date):
    age_days = max(0, int((pd.Timestamp.now(tz="UTC") - latest_data_date).total_seconds() // 86400))
    st.info(f"Opta S3 match data through **{latest_data_date.strftime('%d %b %Y')}** · {age_days} days old. The app uses the latest locally synced S3 dataset.")
    if age_days > 14:
        st.warning("The S3 feed is more than two weeks behind today. Form and predictions cannot reflect matches that have not been added to the bucket.")

with st.sidebar:
    st.header("Player lookup")
    comps = ["All"] + sorted(df["competition"].dropna().unique().tolist())
    competition = st.selectbox("Competition", comps)
    scope = df if competition == "All" else df[df["competition"] == competition]
    options = (scope[["player_id", "player_name", "team"]].drop_duplicates("player_id")
               .assign(label=lambda x: x.player_name + " · " + x.team)
               .sort_values("player_name"))
    selected_label = st.selectbox("Player", options["label"].tolist())
    selected_id = options.loc[options.label == selected_label, "player_id"].iloc[0]
    decay = st.slider("Recency decay λ", 0.05, 0.60, 0.22, 0.01,
                      help="w = exp(-λd), where d is appearances before the latest match.")
    next_minutes = st.slider("Assumed minutes next appearance", 15, 120, 90, 5,
                             help="Used only to project the formula score after the ML forecast.")

player = scope[scope["player_id"] == selected_id].sort_values(["match_date", "match_id"])
if player.empty:
    st.info("No appearances are available for this player in the selected competition.")
    st.stop()

k_consistency = calibrate_consistency_k(df)
summary = current_summary(player, decay=decay, k=k_consistency)
st.subheader(f"{player.iloc[-1]['player_name']} · {player.iloc[-1]['team']}")
st.caption(f"{player.iloc[-1]['position']} · {player.iloc[-1]['competition']} · {len(player)} appearances in this view")

cards = st.columns(5)
cards[0].metric("Current form", f"{summary['overall_form']:.0f} / 100", f"{summary['trend_points']:+.1f} pts vs prior 5")
cards[1].metric("Trend", summary["trend"])
cards[2].metric("Consistency", f"{summary['consistency']:.0f} / 100", "last 5 appearances")
cards[3].metric("Vs historical average", f"{summary['vs_history']:+.1f} pts", f"historical {summary['historical_average']:.1f}")
cards[4].metric("Confidence", f"{summary['confidence']:.0%}", f"{summary['days_since_last_appearance']:.0f} days since appearance")

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Player Form Rating Engine")
    st.write(f"F5 **{summary['form_5']:.1f}** · F10 **{summary['form_10']:.1f}** · F15 **{summary['form_15']:.1f}**")
    st.caption(f"Available sample: {len(player)} appearances · F5/F10/F15 use up to {min(5, len(player))}/{min(10, len(player))}/{min(15, len(player))} appearances.")
    chart = player.tail(15).set_index("match_date")[["match_score"]].rename(columns={"match_score": "Match score"})
    chart_spec = {
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "match_date", "type": "temporal", "title": "Match date"},
            "y": {"field": "Match score", "type": "quantitative", "title": "Score", "scale": {"domain": [0, 100]}},
            "tooltip": [
                {"field": "match_date", "type": "temporal", "title": "Match date"},
                {"field": "Match score", "type": "quantitative", "format": ".1f"},
            ],
        },
    }
    st.vega_lite_chart(chart.reset_index(), chart_spec, height=260, use_container_width=True)
    st.caption("Match scores are position and competition-season relative.")
with right:
    st.subheader("ML Prediction Engine")
    model, training_rows, model_metrics = fit_model(str(DATA), DATA.stat().st_mtime, k_consistency)
    if model is None:
        st.info(model_metrics.get("error", "The model needs more historical appearances."))
    else:
        view_train = training_rows[training_rows["player_id"] == selected_id]
        if len(player) < 2 or view_train.empty:
            st.info("At least two appearances are needed to create this player's history features.")
        else:
            x = latest_feature_row(player, k=k_consistency)
            if x.empty:
                st.info("Not enough history to generate a prediction.")
            else:
                pred = float(model.predict(x)[0])
                projected = project_next_form(player, pred, next_minutes, decay, k_consistency)
                gap = projected - summary["overall_form"]
                direction = "Improving" if gap >= 4 else ("Declining" if gap <= -4 else "Stable")
                st.metric("Projected form after next appearance", f"{projected:.0f} / 100", f"{gap:+.1f} pts vs current form")
                st.write(f"Predicted next-match performance: **{pred:.0f} / 100** · expected trend: **{direction}**")
        st.caption(f"Chronological holdout · model MAE {model_metrics['MAE']:.1f} vs recent-form baseline {model_metrics['baseline_MAE']:.1f} points · R² {model_metrics['R2']:.2f} · {model_metrics['test_rows']} test transitions")
        if model_metrics["MAE_gain"] <= 0:
            st.warning("The ML model did not beat the formula-only recent-form baseline on this holdout. Treat its forecast as experimental.")
        elif model_metrics["R2"] < .10:
            st.warning("The ML model beats the simple baseline slightly, but explains little variation. Treat its forecast as experimental.")
        st.caption(f"The model forecasts the next match score; the form engine rolls that score forward using the assumed {next_minutes} minutes. It is a scenario, not a guaranteed outcome.")

st.subheader("xPoints Engine · team context")
st.caption("Team expected league points (0–3) from independent Poisson goal rates. The supplied S3 files have no xG or attack-level event feed, so this is a transparent recent-goals estimate, not the article's xDanger model or shot-based xG. It is team context, not individual fantasy points.")
latest_player_row = player.iloc[-1]
history_filter = pd.Series(True, index=xpoints_matches.index)
for field in ("competition", "season", "team"):
    if field in xpoints_matches.columns:
        history_filter &= xpoints_matches[field].astype(str).eq(str(latest_player_row[field]))
team_history = xpoints_matches.loc[history_filter].sort_values("match_date")
outlook = project_team_xpoints(team_history)
recent_xp = team_history.dropna(subset=["xpoints"]).tail(5)
xp_cards = st.columns(4)
xp_cards[0].metric("Next-match team xPoints", f"{outlook['xpoints']:.2f} / 3" if pd.notna(outlook["xpoints"]) else "Unavailable", f"from {outlook['matches']} recent team matches" if outlook["matches"] else None)
xp_cards[1].metric("Team goal rate", f"{outlook['goal_rate_for']:.2f}" if pd.notna(outlook["goal_rate_for"]) else "Unavailable", "weighted goals scored per match · last 5")
xp_cards[2].metric("Opponent goal rate", f"{outlook['goal_rate_against']:.2f}" if pd.notna(outlook["goal_rate_against"]) else "Unavailable", "weighted goals conceded per match · last 5")
xp_cards[3].metric("Recent xPoints projections", f"{recent_xp['xpoints'].mean():.2f} / 3" if not recent_xp.empty else "Unavailable", "mean of last 5 team matches")
if not recent_xp.empty:
    st.dataframe(recent_xp[["match_date", "team", "opponent", "goal_rate_for", "goal_rate_against", "p_win", "p_draw", "p_loss", "xpoints"]].sort_values("match_date", ascending=False), use_container_width=True, hide_index=True,
                 column_config={"match_date": st.column_config.DateColumn("Match date"), "goal_rate_for": st.column_config.NumberColumn("Goals scored / match", format="%.2f"), "goal_rate_against": st.column_config.NumberColumn("Goals conceded / match", format="%.2f"), "p_win": st.column_config.NumberColumn("Win prob.", format="%.0%%"), "p_draw": st.column_config.NumberColumn("Draw prob.", format="%.0%%"), "p_loss": st.column_config.NumberColumn("Loss prob.", format="%.0%%"), "xpoints": st.column_config.NumberColumn("xPoints", format="%.2f")})

st.subheader("Recent appearances")
show_cols = ["match_date", "competition", "season", "team", "opponent", "minutes", "match_score", "goals", "assists", "shots", "shots_on_target", "tackles", "tackles_won", "clearances", "pass_accuracy"]
recent = player.tail(10).copy()
recent["opponent"] = recent.apply(lambda r: r["away_team"] if r["team"] == r["home_team"] else r["home_team"], axis=1)
recent["pass_accuracy"] = (recent["pass_accuracy"] * 100).round(1)
recent["match_date"] = recent["match_date"].dt.strftime("%Y-%m-%d")
st.dataframe(recent[show_cols].sort_values("match_date", ascending=False), use_container_width=True, hide_index=True)

with st.expander("Formula and data limitations"):
    st.markdown(r"""
**Recent form:** `Form = 100 × Σ(wᵢPᵢ) / Σwᵢ`, with `wᵢ = exp(−λdᵢ) × √(minutesᵢ/90)`. F5, F10 and F15 are combined as `0.50 F5 + 0.30 F10 + 0.20 F15`.

**Confidence:** Minutes confidence is `min(1, √(minutes in last 15 appearances / 450))`. It is multiplied by a calendar freshness factor `exp(−max(0, days_since_last_appearance − 7)/90)`; adjusted form is `50 + confidence × (raw form − 50)`. This reduces certainty after long breaks, but does not label their cause.

**Match score:** weighted percentiles for role-relevant per-90 stats within position, competition and season. Higher is better; adverse events are reverse ranked. Consistency is `100 − min(100, k × sample SD)` for the last five scores, where `k` is calibrated so the dataset median five-match dispersion maps to a 30-point deduction.

**Prediction:** Random Forest regression uses only earlier appearances to predict the next match-score percentile. Its chronological holdout is compared with the recent-form formula. The S3 matchstats feed supplies stable Opta IDs, minutes, goals, assists, shots, passing, tackles, clearances and cards; it has no xG, xA, injury status, interceptions or recoveries.

**Breaks and availability:** `days_rest` is the calendar-day gap between a player's appearances and is a model feature. Long gaps can mean injury, rotation, suspension, transfer or missing feed data; this source cannot distinguish them. The app does not claim a player is injured without explicit injury records.

**xPoints:** The article calculates expected points from goal probabilities using `3 × P(win) + P(draw)`. The S3 files do not include shot-level xG or attack sequences, so this app estimates goal rates from recent team goals and applies independent Poisson score probabilities. It is not the article's xDanger calculation. [Marc Lamberts' xDanger article](https://marclamberts.medium.com/introducing-xdanger-one-ecosystem-for-shot-models-possession-value-models-and-ratings-a00f51663a1d).
""")
    st.caption("Data: Opta S3 matchstats in the provided buckets; no football-data API calls are made.")

st.caption(f"Dataset file: {DATA.relative_to(ROOT)}")
