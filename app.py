from pathlib import Path

import pandas as pd
import streamlit as st

from engines import calibrate_consistency_k, current_summary, latest_feature_row, prepare_match_scores, project_next_form, train_predictor

ROOT = Path(__file__).resolve().parent
API_DATA = ROOT / "data" / "processed" / "player_match_stats_api.csv"
LEGACY_DATA = ROOT / "data" / "processed" / "player_match_stats.csv"
DATA = API_DATA if API_DATA.exists() else LEGACY_DATA

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
    st.code("python scripts/sync_pitchapi.py", language="powershell")
    st.markdown("Add a free PitchAPI token to `.env` first. The importer caches each finished match's player stats and writes the current-season dataset.")
    st.stop()

df = load_data(str(DATA), DATA.stat().st_mtime)
if df.empty:
    st.error("The dataset is empty. Rebuild it with `python scripts/build_dataset.py`.")
    st.stop()
latest_data_date = df["match_date"].max()
if pd.notna(latest_data_date):
    if DATA == API_DATA:
        st.info(f"PitchAPI match data through **{latest_data_date.strftime('%d %b %Y')}**. Scores reflect the latest finished match stats synced locally; run the sync script after match days to refresh.")
    else:
        st.warning(f"This is the old archived dataset through **{latest_data_date.strftime('%d %b %Y')}**. Add a PitchAPI key and run the sync script for 2026–27 data.")

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
cards[4].metric("Confidence", f"{summary['confidence']:.0%}", f"{summary['minutes_last_15']:.0f} minutes in last 15")

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

st.subheader("Recent appearances")
show_cols = ["match_date", "competition", "season", "team", "opponent", "minutes", "match_score", "goals", "assists", "xg", "xa", "shots", "tackles", "recoveries", "pass_accuracy"]
recent = player.tail(10).copy()
recent["opponent"] = recent.apply(lambda r: r["away_team"] if r["team"] == r["home_team"] else r["home_team"], axis=1)
recent["pass_accuracy"] = (recent["pass_accuracy"] * 100).round(1)
recent["match_date"] = recent["match_date"].dt.strftime("%Y-%m-%d")
st.dataframe(recent[show_cols].sort_values("match_date", ascending=False), use_container_width=True, hide_index=True)

with st.expander("Formula and data limitations"):
    st.markdown(r"""
**Recent form:** `Form = 100 × Σ(wᵢPᵢ) / Σwᵢ`, with `wᵢ = exp(−λdᵢ) × √(minutesᵢ/90)`. F5, F10 and F15 are combined as `0.50 F5 + 0.30 F10 + 0.20 F15`.

**Confidence:** `min(1, √(minutes in last 15 appearances / 450))`; adjusted form is `50 + confidence × (raw form − 50)`.

**Match score:** weighted percentiles for role-relevant per-90 stats within position, competition and season. Higher is better; adverse events are reverse ranked. Consistency is `100 − min(100, k × sample SD)` for the last five scores, where `k` is calibrated so the dataset median five-match dispersion maps to a 30-point deduction.

**Prediction:** Random Forest regression uses only previous appearances to predict the next match's score. The chronological holdout is compared with the recent-form formula as a simple baseline. xG/xA and defensive event coverage depend on what StatsBomb recorded; opponent strength, team form and injury context are not included in this first version. StatsBomb IDs are source-specific, and the dataset covers selected competitions only.
""")
    st.caption("Data: StatsBomb Open Data. Credit StatsBomb when sharing analysis or results.")

st.caption("Dataset file: data/processed/player_match_stats.csv")
