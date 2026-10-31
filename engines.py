from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

WINDOWS = (5, 10, 15)
WEIGHTS = {
    "Goalkeeper": {"saves_p90": .28, "pass_accuracy": .16, "recoveries_p90": .12, "clearances_p90": .12, "pressures_p90": .08, "shots_on_target_p90": .08, "xg_p90": .08, "xa_p90": .08},
    "Defender": {"tackles_won_p90": .16, "interceptions_p90": .14, "clearances_p90": .14, "blocks_p90": .10, "duels_won_p90": .10, "aerials_won_p90": .06, "pass_accuracy": .12, "recoveries_p90": .08, "xg_p90": .04, "xa_p90": .03, "dispossessed_p90": -.04, "fouls_p90": -.04},
    "Defensive Midfielder": {"tackles_won_p90": .15, "interceptions_p90": .14, "recoveries_p90": .12, "duels_won_p90": .08, "pass_accuracy": .16, "xa_p90": .08, "key_passes_p90": .06, "carries_p90": .07, "pressures_p90": .08, "dispossessed_p90": -.03, "fouls_p90": -.03},
    "Midfielder": {"pass_accuracy": .17, "xa_p90": .12, "key_passes_p90": .10, "tackles_won_p90": .10, "interceptions_p90": .08, "recoveries_p90": .08, "duels_won_p90": .07, "carries_p90": .09, "pressures_p90": .08, "xg_p90": .06, "dispossessed_p90": -.03, "fouls_p90": -.02},
    "Forward": {"goals_p90": .20, "xg_p90": .16, "shots_on_target_p90": .10, "xa_p90": .10, "assists_p90": .08, "successful_dribbles_p90": .08, "key_passes_p90": .08, "pressures_p90": .06, "pass_accuracy": .05, "carries_p90": .05, "dispossessed_p90": -.02, "fouls_p90": -.02},
    "Unknown": {"goals_p90": .14, "xg_p90": .12, "xa_p90": .10, "pass_accuracy": .12, "tackles_won_p90": .10, "interceptions_p90": .10, "recoveries_p90": .10, "pressures_p90": .08, "saves_p90": .08, "dispossessed_p90": -.03, "fouls_p90": -.03},
}
COUNT_METRICS = ["goals", "assists", "shots", "shots_on_target", "passes", "accurate_passes", "key_passes", "carries", "dribbles", "successful_dribbles", "pressures", "interceptions", "clearances", "recoveries", "tackles", "tackles_won", "duels_won", "aerials", "aerials_won", "blocks", "fouls", "miscontrols", "dispossessed", "saves", "xg", "xa"]


def prepare_match_scores(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce", utc=True)
    for c in COUNT_METRICS + ["minutes"]:
        if c not in df: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["position"] = df["position"].fillna("Unknown")
    mins = df["minutes"].clip(lower=1)
    for metric in COUNT_METRICS:
        df[f"{metric}_p90"] = df[metric] * 90 / mins
    df["pass_accuracy"] = np.where(df["passes"] > 0, df["accurate_passes"] / df["passes"], 0)
    # Goals conceded is not supplied consistently in this event subset; do not invent it.
    keys = [k for k in ["competition", "season", "position"] if k in df]
    scores = []
    for _, group in df.groupby(keys, dropna=False, sort=False):
        g = group.copy()
        for col in set(m for role in WEIGHTS.values() for m in role):
            if col not in g: g[col] = 0.0
            g[col + "_pct"] = g[col].rank(method="average", pct=True) * 100
        row_scores = []
        for ix, r in g.iterrows():
            weights = WEIGHTS.get(r["position"], WEIGHTS["Unknown"])
            num = den = 0.0
            for metric, weight in weights.items():
                # Negative coefficients use the reverse percentile direction.
                pct = float(r.get(metric + "_pct", 50.0))
                val = pct if weight > 0 else 100 - pct
                magnitude = abs(weight)
                num += val * magnitude
                den += magnitude
            row_scores.append((ix, num / den if den else 50.0))
        scores.extend(row_scores)
    score_map = dict(scores)
    df["match_score"] = pd.Series(df.index.map(score_map), index=df.index, dtype=float).clip(1, 100)
    df = df.sort_values(["player_id", "match_date", "match_id"]).reset_index(drop=True)
    return df


def recency_form(scores, minutes, decay: float = .22) -> float:
    p = np.asarray(scores, dtype=float)
    m = np.asarray(minutes, dtype=float)
    if not len(p): return 50.0
    # Arrays are ordered oldest -> newest; d=0 is the latest appearance.
    d = np.arange(len(p) - 1, -1, -1)
    w = np.exp(-decay * d) * np.sqrt(np.clip(m, 0, 90) / 90)
    return float(np.average(p, weights=w)) if w.sum() else 50.0


def consistency_score(scores, k: float = .65) -> float:
    if len(scores) < 2: return 50.0
    return float(np.clip(100 - min(100, k * np.std(scores, ddof=1)), 0, 100))


def calibrate_consistency_k(scored: pd.DataFrame) -> float:
    dispersions = []
    for _, player in scored.groupby("player_id", sort=False):
        s = player.sort_values("match_date")["match_score"].tail(5)
        if len(s) >= 2: dispersions.append(float(s.std(ddof=1)))
    median = float(np.median(dispersions)) if dispersions else 30.0
    return 30.0 / max(median, 1.0)


def current_summary(player: pd.DataFrame, decay: float = .22, k: float = .65) -> dict:
    player = player.sort_values(["match_date", "match_id"])
    scores = player["match_score"].to_numpy(float)
    minutes = player["minutes"].to_numpy(float)
    fn = {n: recency_form(scores[-n:], minutes[-n:], decay) for n in WINDOWS}
    combined = .50 * fn[5] + .30 * fn[10] + .20 * fn[15]
    total_mins = float(minutes[-15:].sum())
    confidence = min(1.0, np.sqrt(total_mins / 450))
    adjusted = float(np.clip(50 + confidence * (combined - 50), 1, 100))
    last5 = scores[-5:]
    prior5 = scores[-10:-5]
    trend = float(np.mean(last5) - np.mean(prior5)) if len(prior5) else 0.0
    hist_avg = float(np.mean(scores)) if len(scores) else 50.0
    return {"form_5": fn[5], "form_10": fn[10], "form_15": fn[15], "overall_form": adjusted,
            "raw_overall_form": combined, "confidence": confidence, "consistency": consistency_score(last5, k),
            "trend_points": trend, "trend": "Improving" if trend >= 4 else ("Declining" if trend <= -4 else "Stable"),
            "historical_average": hist_avg, "vs_history": adjusted - hist_avg,
            "appearances": len(player), "minutes_last_15": total_mins}


def project_next_form(player: pd.DataFrame, predicted_score: float, next_minutes: float, decay: float = .22, k: float = .65) -> float:
    p = player.sort_values(["match_date", "match_id"])
    scores = np.append(p["match_score"].to_numpy(float), float(predicted_score))
    minutes = np.append(p["minutes"].to_numpy(float), float(next_minutes))
    forms = {n: recency_form(scores[-n:], minutes[-n:], decay) for n in WINDOWS}
    raw = .50 * forms[5] + .30 * forms[10] + .20 * forms[15]
    confidence = min(1.0, np.sqrt(float(minutes[-15:].sum()) / 450))
    return float(np.clip(50 + confidence * (raw - 50), 1, 100))


FEATURES = ["form5", "form10", "form15", "consistency", "trend_points", "minutes5", "minutes10", "appearances", "goals5", "goals10", "xg5", "xg10", "xa5", "xa10", "shots5", "tackles5", "recoveries5", "pass_accuracy5", "days_rest"]


def make_training_rows(scored: pd.DataFrame, k: float = .65) -> pd.DataFrame:
    rows = []
    for pid, player in scored.groupby("player_id", sort=False):
        p = player.sort_values(["match_date", "match_id"]).reset_index(drop=True)
        for i in range(1, len(p)):
            hist = p.iloc[:i]
            target = p.iloc[i]
            row = {"player_id": pid, "target_date": target["match_date"], "target_score": float(target["match_score"]), "position": target["position"]}
            scores = hist["match_score"].to_numpy(float)
            mins = hist["minutes"].to_numpy(float)
            forms = {n: recency_form(scores[-n:], mins[-n:]) for n in WINDOWS}
            row.update({"form5": forms[5], "form10": forms[10], "form15": forms[15], "consistency": consistency_score(scores[-5:], k), "trend_points": float(scores[-5:].mean() - scores[-10:-5].mean()) if len(scores) >= 6 else 0.0,
                        "minutes5": float(mins[-5:].sum()), "minutes10": float(mins[-10:].sum()), "appearances": i})
            for metric, col in [("goals", "goals"), ("xg", "xg"), ("xa", "xa"), ("shots", "shots"), ("tackles", "tackles"), ("recoveries", "recoveries")]:
                row[f"{col}5"] = float(hist[metric].tail(5).sum())
                if col in {"goals", "xg", "xa"}: row[f"{col}10"] = float(hist[metric].tail(10).sum())
            ps = hist["passes"].tail(5).sum()
            row["pass_accuracy5"] = float(hist["accurate_passes"].tail(5).sum() / ps) if ps else 0.0
            if pd.notna(target["match_date"]) and pd.notna(hist.iloc[-1]["match_date"]):
                row["days_rest"] = max(0, (target["match_date"] - hist.iloc[-1]["match_date"]).total_seconds() / 86400)
            else: row["days_rest"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def train_predictor(scored: pd.DataFrame, k: float = .65):
    data = make_training_rows(scored, k).dropna(subset=FEATURES + ["target_score", "target_date"])
    if len(data) < 30:
        return None, data, {"error": "At least 30 player appearance transitions are needed to train."}
    data = data.sort_values("target_date")
    cutoff = data["target_date"].quantile(.8)
    train = data[data["target_date"] <= cutoff]
    test = data[data["target_date"] > cutoff]
    if len(test) < 5 or len(train) < 20:
        return None, data, {"error": "Not enough appearances on both sides of the chronological holdout."}
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=3, max_features=.8, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["target_score"])
    pred = np.clip(model.predict(test[FEATURES]), 0, 100)
    baseline = test["form5"].to_numpy(float)
    model_mae = float(mean_absolute_error(test["target_score"], pred))
    baseline_mae = float(mean_absolute_error(test["target_score"], baseline))
    metrics = {"MAE": model_mae, "baseline_MAE": baseline_mae, "MAE_gain": baseline_mae - model_mae,
               "R2": float(r2_score(test["target_score"], pred)), "train_rows": len(train), "test_rows": len(test), "cutoff": str(cutoff.date())}
    # Refit on all rows for use inside the app after reporting holdout metrics.
    model.fit(data[FEATURES], data["target_score"])
    return model, data, metrics


def latest_feature_row(player: pd.DataFrame, k: float = .65) -> pd.DataFrame:
    p = player.sort_values(["match_date", "match_id"])
    # Add a synthetic future target so all observed appearances become past-only features.
    if p.empty: return pd.DataFrame()
    future = p.iloc[-1].copy()
    future["match_date"] = p.iloc[-1]["match_date"] + pd.Timedelta(days=7)
    # The synthetic row is seven days after the last appearance, so its ID is
    # only a stable tie-breaker. Preserve the provider's string ID type here.
    future["match_id"] = p.iloc[-1]["match_id"]
    all_rows = make_training_rows(pd.concat([p, pd.DataFrame([future])], ignore_index=True), k)
    if all_rows.empty: return pd.DataFrame()
    return all_rows.iloc[[-1]][FEATURES]
