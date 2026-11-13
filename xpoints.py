"""Team expected-points calculations adapted from McKay Johns' xPts method.

The source method predicts team league points from both teams' xG using
independent Poisson goal distributions. It does not define player xPoints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _goal_probabilities(xg: float, max_goals: int = 10) -> np.ndarray:
    """Poisson probabilities for 0..max_goals, retaining tail mass at max."""
    rate = float(np.clip(xg, 0.0, 8.0))
    probs = np.empty(max_goals + 1, dtype=float)
    probs[0] = np.exp(-rate)
    for goals in range(1, max_goals + 1):
        probs[goals] = probs[goals - 1] * rate / goals
    # Fold omitted high-score probability into the last bucket.
    probs[-1] += max(0.0, 1.0 - probs.sum())
    return probs


def expected_points(xg_for: float, xg_against: float) -> dict[str, float]:
    """Return W/D/L probabilities and expected league points (0..3)."""
    if pd.isna(xg_for) or pd.isna(xg_against):
        return {"p_win": np.nan, "p_draw": np.nan, "p_loss": np.nan, "xpoints": np.nan}
    matrix = np.outer(_goal_probabilities(xg_for), _goal_probabilities(xg_against))
    p_win = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_loss = float(np.triu(matrix, 1).sum())
    return {"p_win": p_win, "p_draw": p_draw, "p_loss": p_loss,
            "xpoints": float(3 * p_win + p_draw)}


def team_match_xpoints(matches: pd.DataFrame) -> pd.DataFrame:
    """Build one row per team-match from player xG totals in the dataset."""
    needed = {"match_id", "team", "opponent", "xg"}
    if not needed.issubset(matches.columns):
        return pd.DataFrame(columns=["match_id", "team", "opponent", "xg_for", "xg_against", "xpoints"])
    keys = [c for c in ("competition", "season", "match_id", "match_date", "team", "opponent")
            if c in matches.columns]
    base_keys = [c for c in ("competition", "season", "match_id", "team") if c in keys]
    work = matches.copy()
    work["xg"] = pd.to_numeric(work["xg"], errors="coerce")
    # Groupby sum(min_count=1) preserves unavailable xG as missing.
    team = work.groupby(base_keys, dropna=False, as_index=False)["xg"].sum(min_count=1)
    names = [c for c in ("match_date", "opponent") if c in work.columns]
    if names:
        meta = work.groupby(base_keys, dropna=False, as_index=False)[names].first()
        team = team.merge(meta, on=base_keys, how="left")
    team = team.rename(columns={"xg": "xg_for"})

    # Pair the opposing team's xG within the same competition-season-match.
    pair_keys = [c for c in ("competition", "season", "match_id") if c in team.columns]
    against = team[pair_keys + ["team", "xg_for"]].rename(
        columns={"team": "opponent", "xg_for": "xg_against"})
    team = team.merge(against, on=pair_keys + ["opponent"], how="left")
    vals = [expected_points(a, b) for a, b in zip(team["xg_for"], team["xg_against"])]
    team[["p_win", "p_draw", "p_loss", "xpoints"]] = pd.DataFrame(vals, index=team.index)
    return team


def project_team_xpoints(team_history: pd.DataFrame, window: int = 5) -> dict[str, float | int]:
    """Project the next match from recency-weighted recent team xG for/against."""
    hist = team_history.dropna(subset=["xg_for", "xg_against"]).sort_values("match_date").tail(window)
    if hist.empty:
        return {"xg_for": np.nan, "xg_against": np.nan, "xpoints": np.nan, "matches": 0}
    # Most recent match gets the largest weight, with a modest exponential decay.
    distances = np.arange(len(hist) - 1, -1, -1, dtype=float)
    weights = np.exp(-0.30 * distances)
    xgf = float(np.average(hist["xg_for"].astype(float), weights=weights))
    xga = float(np.average(hist["xg_against"].astype(float), weights=weights))
    return {**expected_points(xgf, xga), "xg_for": xgf, "xg_against": xga, "matches": len(hist)}
