"""Incrementally import match-level player statistics from PitchAPI.

Default scope is Premier League 2025/2026 + 2026/2027. JSON responses are
cached per match, so repeat runs spend API calls only on newly finished games.
Set PITCHAPI_API_KEY in the environment or project .env before running.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pitchapi"
PROCESSED = ROOT / "data" / "processed"
BASE_URL = "https://api.pitchapi.dev/v1"
LEAGUE_NAME = os.getenv("PITCHAPI_LEAGUE", "Premier League")
COUNTRY_CODE = os.getenv("PITCHAPI_COUNTRY_CODE", "ENG").upper()
SEASONS = [s.strip() for s in os.getenv("PITCHAPI_SEASONS", "2025/2026,2026/2027").split(",") if s.strip()]
REQUEST_PAUSE = float(os.getenv("PITCHAPI_REQUEST_PAUSE", "0.35"))

STAT_ALIASES = {
    "goals": "goals", "assists": "assists", "expected_goals": "xg", "xg": "xg",
    "xag": "xa", "expected_assisted_goals": "xa", "expected_assists": "xa", "total_shots": "shots", "shots": "shots",
    "shots_on_target": "shots_on_target", "total_passes": "passes", "passes": "passes",
    "accurate_passes": "accurate_passes", "pass_accuracy": "pass_accuracy", "key_passes": "key_passes",
    "crosses": "crosses", "tackles": "tackles", "tackles_won": "tackles_won",
    "interceptions": "interceptions", "clearances": "clearances", "blocks": "blocks",
    "duels_won": "duels_won", "aerials": "aerials", "aerials_won": "aerials_won",
    "recoveries": "recoveries", "saves": "saves", "fouls": "fouls",
    "yellow_cards": "yellow_cards", "red_cards": "red_cards", "minutes_played": "minutes",
    "total_dribbles": "dribbles", "successful_dribbles": "successful_dribbles",
    "dribbles_won": "successful_dribbles", "dispossessed": "dispossessed", "miscontrols": "miscontrols",
    "duel_won": "duels_won", "shots_off_target": "shots_off_target", "goalsprevented": "goals_prevented",
    "goals_conceded": "goals_conceded", "passes_into_final_third": "passes_into_final_third",
    "accurate_crosses": "accurate_crosses", "dribbles_succeeded": "successful_dribbles",
    "matchstats.headers.tackles": "tackles", "offsides": "offsides",
}
NORMALIZED_ALIASES = {re.sub(r"[^a-z0-9]", "", key.casefold()): value for key, value in STAT_ALIASES.items()}


def load_project_env():
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("PITCHAPI_API_KEY="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            os.environ.setdefault("PITCHAPI_API_KEY", value)


class PitchClient:
    def __init__(self, key: str):
        self.session = requests.Session()
        self.session.headers.update({"X-API-KEY": key, "Accept": "application/json", "User-Agent": "football-player-form/1.0"})
        self.last_request = 0.0

    def get(self, path: str, params=None):
        delay = REQUEST_PAUSE - (time.monotonic() - self.last_request)
        if delay > 0:
            time.sleep(delay)
        response = self.session.get(f"{BASE_URL}{path}", params=params or {}, timeout=45)
        self.last_request = time.monotonic()
        if response.status_code == 429:
            retry = float(response.headers.get("Retry-After", "5"))
            time.sleep(min(max(retry, 1), 90))
            response = self.session.get(f"{BASE_URL}{path}", params=params or {}, timeout=45)
            self.last_request = time.monotonic()
        if response.status_code in (401, 403):
            raise RuntimeError("PitchAPI rejected the key. Check PITCHAPI_API_KEY; the key value is never printed.")
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"PitchAPI error: {payload['error']}")
        return payload.get("data", payload)


def cached_player_stats(client: PitchClient, match_id: str):
    path = RAW / "players" / f"{safe(match_id)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = client.get(f"/matches/{match_id}/players")
    entries = data if isinstance(data, list) else data.get("players", []) if isinstance(data, dict) else []
    if not entries:
        # Keep unready matches uncached so a later sync can pick up the post-match stats.
        return data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def cached_lineups(client: PitchClient, match_id: str):
    path = RAW / "lineups" / f"{safe(match_id)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = client.get(f"/matches/{match_id}/lineups")
    if not isinstance(data, dict) or not (data.get("home") or data.get("away")):
        return data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def stat_value(raw: dict):
    val = raw.get("value")
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try: return float(val.replace("%", ""))
        except ValueError: return None
    return None


def position_group(position_id):
    """Map PitchAPI lineup role codes to broad football roles for normalization."""
    try:
        code = int(position_id)
    except (TypeError, ValueError):
        return "Unknown"
    if code in (1, 11):
        return "Goalkeeper"
    if code == 2 or 20 <= code < 40:
        return "Defender"
    if code == 3 or 40 <= code < 100:
        return "Midfielder"
    if code == 4 or code >= 100:
        return "Forward"
    return "Unknown"


def flatten_player(match: dict, league_name: str, season: str, item: dict):
    player = item.get("player") or {}
    pid = player.get("id")
    lineup_positions = item.get("_lineup_positions", {})
    team_id = item.get("team_id")
    home = match.get("home_team") or {}
    away = match.get("away_team") or {}
    team = home if team_id == home.get("id") else away
    opponent = away if team_id == home.get("id") else home
    stats = {}
    for group in item.get("stats") or []:
        for label, entry in (group.get("stats") or {}).items():
            raw_key = entry.get("key") or label
            key = re.sub(r"[^a-z0-9]", "", raw_key.casefold())
            canonical = NORMALIZED_ALIASES.get(key)
            if not canonical:
                continue
            detail = entry.get("stat") or {}
            value = stat_value(detail)
            if value is not None:
                stats[canonical] = value
            if canonical == "accurate_passes" and detail.get("total") is not None:
                stats["passes"] = float(detail["total"])
    pos_id = player.get("position_id") or lineup_positions.get(pid)
    position = position_group(pos_id)
    row = {
        "match_id": match.get("id"), "match_date": match.get("time_utc") or match.get("date"),
        "competition": league_name, "season": season, "home_team": home.get("name"),
        "away_team": away.get("name"), "team": team.get("name"), "opponent": opponent.get("name"),
        "player_id": pid, "player_name": player.get("name"), "position_id": pos_id,
        "position": position, "minutes": stats.pop("minutes", 0.0),
    }
    row.update(stats)
    # Keep available stats numeric and missing metrics absent; the rating engine
    # interprets absent additive counts as zero and calculates accuracy if possible.
    if "passes" in row and "accurate_passes" in row:
        row["pass_accuracy"] = row["accurate_passes"] / row["passes"] if row["passes"] else 0.0
    return row if pid is not None and row["minutes"] > 0 else None


def main():
    load_project_env()
    key = os.getenv("PITCHAPI_API_KEY", "").strip()
    if not key:
        raise SystemExit("Add PITCHAPI_API_KEY to football_player_form/.env (see .env.example), then rerun this command.")
    RAW.mkdir(parents=True, exist_ok=True)
    client = PitchClient(key)
    leagues_data = client.get("/leagues")
    leagues = leagues_data.get("leagues", []) if isinstance(leagues_data, dict) else leagues_data
    candidates = [x for x in leagues if x.get("name", "").casefold() == LEAGUE_NAME.casefold()]
    league = next((x for x in candidates if x.get("country_code", "").upper() == COUNTRY_CODE), None)
    if league is None and candidates:
        league = candidates[0]
    if not league:
        available = ", ".join(x.get("name", "") for x in leagues)
        raise SystemExit(f"League '{LEAGUE_NAME}' was not found. Available: {available}")
    available_seasons = league.get("seasons", [])
    missing = [season for season in SEASONS if season not in available_seasons]
    if missing:
        raise SystemExit(f"PitchAPI does not list requested season(s) {missing} for {LEAGUE_NAME}. Available: {available_seasons}")

    all_matches = []
    for season in SEASONS:
        result = client.get(f"/leagues/{league['id']}/matches", {"season": season})
        matches = result.get("matches", []) if isinstance(result, dict) else result
        all_matches.extend((league["name"], season, m) for m in matches if m.get("status", "").casefold() == "finished")

    rows = []
    match_count = 0
    for competition, season, match in all_matches:
        payload = cached_player_stats(client, str(match["id"]))
        entries = payload if isinstance(payload, list) else payload.get("players", []) if isinstance(payload, dict) else []
        positions = {}
        if any(not (item.get("player") or {}).get("position_id") for item in entries):
            lineup_data = cached_lineups(client, str(match["id"]))
            for side in ("home", "away"):
                for group in (lineup_data.get(side) or {}).values() if isinstance(lineup_data, dict) else []:
                    if isinstance(group, list):
                        for person in group:
                            if person.get("player_id") and person.get("position_id"):
                                positions[person["player_id"]] = person["position_id"]
        for item in entries:
            item["_lineup_positions"] = positions
            row = flatten_player(match, competition, season, item)
            if row:
                rows.append(row)
        match_count += 1
        if match_count % 25 == 0 or match_count == len(all_matches):
            print(f"Imported player stats for {match_count}/{len(all_matches)} finished matches")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No player appearances were returned; check the league/season coverage in the API response.")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce", utc=True)
    df = df.sort_values(["player_id", "match_date", "match_id"]).drop_duplicates(["match_id", "player_id"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    all_path = PROCESSED / "player_match_stats_api.csv"
    df.to_csv(all_path, index=False)
    current_season = SEASONS[-1]
    current = df[df["season"] == current_season]
    current_path = PROCESSED / f"player_match_stats_{safe(current_season)}.csv"
    current.to_csv(current_path, index=False)
    print(f"Saved {len(df)} appearances ({df.match_id.nunique()} matches; {df.player_id.nunique()} player IDs) to {all_path}")
    print(f"Saved {len(current)} current-season appearances ({current.match_id.nunique()} matches) to {current_path}")


if __name__ == "__main__":
    main()
