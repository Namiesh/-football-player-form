from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "player_match_stats.csv"
BASE = "https://raw.githubusercontent.com/hudl/open-data/master/data"
# A compact, reproducible subset with repeated league appearances plus tournaments.
SEASONS = [(43, 106), (55, 282)]  # FIFA World Cup 2022, UEFA Euro 2024


def get_json(session: requests.Session, url: str, target: Path):
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    target.write_bytes(response.content)
    # Keep the public host request rate modest; cached files are never requested again.
    time.sleep(0.08)
    return response.json()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "football-player-form-research/1.0"})
    comps = get_json(session, f"{BASE}/competitions.json", RAW / "competitions.json")
    comp_lookup = {(int(c["competition_id"]), int(c["season_id"])): c for c in comps}
    match_rows = []
    for comp_id, season_id in SEASONS:
        match_file = RAW / "matches" / str(comp_id) / f"{season_id}.json"
        matches = get_json(session, f"{BASE}/matches/{comp_id}/{season_id}.json", match_file)
        if matches is None:
            print(f"Missing match file: {comp_id}/{season_id}")
            continue
        meta = comp_lookup.get((comp_id, season_id), {})
        for m in matches:
            m["_competition"] = meta.get("competition_name", str(comp_id))
            m["_season"] = meta.get("season_name", str(season_id))
            match_rows.append(m)

    output = []
    total = len(match_rows)
    for ix, match in enumerate(match_rows, start=1):
        mid = int(match["match_id"])
        events = get_json(session, f"{BASE}/events/{mid}.json", RAW / "events" / f"{mid}.json")
        lineups = get_json(session, f"{BASE}/lineups/{mid}.json", RAW / "lineups" / f"{mid}.json")
        if events is None or lineups is None:
            continue
        output.extend(aggregate_match(match, events, lineups))
        if ix % 50 == 0 or ix == total:
            print(f"Processed {ix}/{total} matches; {len(output)} player appearances")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(output)
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce", utc=True)
        df = df.sort_values(["player_id", "match_date", "match_id"])
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")


def aggregate_match(match, events, lineups):
    match_id = match["match_id"]
    date = match.get("match_date")
    home = match.get("home_team", {}).get("home_team_name", "")
    away = match.get("away_team", {}).get("away_team_name", "")
    players = {}
    event_by_id = {ev.get("id"): ev for ev in events if ev.get("id")}
    for team in lineups:
        for p in team.get("lineup", []):
            pid = p.get("player_id")
            if pid is None:
                continue
            positions = p.get("positions") or []
            position = positions[0].get("position", "Unknown") if positions else "Unknown"
            periods = []
            for pos in positions:
                start = parse_time(pos.get("from"))
                end = parse_time(pos.get("to"))
                if start is not None:
                    periods.append((start, end))
            players[pid] = {
                "player_id": pid,
                "player_name": p.get("player_nickname") or p.get("player_name") or str(pid),
                "team": team.get("team_name", ""),
                "position": normalize_position(position),
                "periods": periods,
                "stats": {k: 0.0 for k in METRICS},
                "xg": 0.0,
                "xa": 0.0,
            }

    max_minute = 90.0
    for ev in events:
        max_minute = max(max_minute, event_minute(ev))
        typ = ev.get("type", {}).get("name", "")
        actor = ev.get("player") or {}
        pid = actor.get("id")
        if pid is None:
            continue
        if pid not in players:
            # Some substitutes appear only in events, so retain them with unknown role.
            players[pid] = {"player_id": pid, "player_name": actor.get("name", str(pid)), "team": (ev.get("team") or {}).get("name", ""), "position": "Unknown", "periods": [], "stats": {k: 0.0 for k in METRICS}, "xg": 0.0, "xa": 0.0}
        row = players[pid]
        s = row["stats"]
        if typ == "Pass":
            s["passes"] += 1
            if not ev.get("pass", {}).get("outcome"):
                s["accurate_passes"] += 1
        elif typ == "Shot":
            shot = ev.get("shot", {})
            s["shots"] += 1
            shot_xg = shot.get("statsbomb_xg", 0) or 0
            row["xg"] += shot_xg
            if shot.get("outcome", {}).get("name") in {"Goal", "Saved", "Post"}:
                s["shots_on_target"] += 1
            if shot.get("outcome", {}).get("name") == "Goal":
                s["goals"] += 1
            key_pass = event_by_id.get(shot.get("key_pass_id"))
            assister = (key_pass or {}).get("player", {})
            assister_id = assister.get("id")
            if assister_id is not None:
                if assister_id not in players:
                    players[assister_id] = {"player_id": assister_id, "player_name": assister.get("name", str(assister_id)), "team": (key_pass.get("team") or {}).get("name", ""), "position": "Unknown", "periods": [], "stats": {k: 0.0 for k in METRICS}, "xg": 0.0, "xa": 0.0}
                players[assister_id]["xa"] += shot_xg
                if shot.get("outcome", {}).get("name") == "Goal":
                    players[assister_id]["stats"]["assists"] += 1
        elif typ == "Carry": s["carries"] += 1
        elif typ == "Dribble":
            s["dribbles"] += 1
            if not ev.get("dribble", {}).get("outcome") or ev["dribble"]["outcome"].get("name") == "Complete": s["successful_dribbles"] += 1
        elif typ == "Pressure": s["pressures"] += 1
        elif typ == "Interception": s["interceptions"] += 1
        elif typ == "Clearance": s["clearances"] += 1
        elif typ == "Ball Recovery": s["recoveries"] += 1
        elif typ == "Duel" and (ev.get("duel", {}).get("type") or {}).get("name") == "Tackle":
            s["tackles"] += 1
            if ev.get("duel", {}).get("outcome", {}).get("name") in {"Won", "Success In Play", "Success Out"}: s["tackles_won"] += 1
        elif typ == "Block": s["blocks"] += 1
        elif typ == "Foul Committed": s["fouls"] += 1
        elif typ == "Miscontrol": s["miscontrols"] += 1
        elif typ == "Dispossessed": s["dispossessed"] += 1
        elif typ == "Goal Keeper":
            if ev.get("goalkeeper", {}).get("type", {}).get("name") == "Shot Saved": s["saves"] += 1
        elif typ == "Substitution":
            off = ev.get("player", {}).get("id")
            on = (ev.get("substitution") or {}).get("replacement", {}).get("id")
            minute = event_minute(ev)
            if off in players:
                players[off]["periods"].append((0.0, minute))
            if on is not None:
                repl = (ev.get("substitution") or {}).get("replacement", {})
                if on not in players:
                    players[on] = {"player_id": on, "player_name": repl.get("name", str(on)), "team": row["team"], "position": "Unknown", "periods": [], "stats": {k: 0.0 for k in METRICS}, "xg": 0.0, "xa": 0.0}
                players[on]["periods"].append((minute, max_minute))

    # Derive appearances and minutes from lineup position intervals; fallback to event activity.
    result = []
    for p in players.values():
        durations = [(max_minute if b is None else b) - a for a, b in p["periods"] if (max_minute if b is None else b) > a]
        mins = sum(durations) if durations else (max_minute if any(p["stats"].values()) else 0)
        if mins <= 0:
            continue
        rec = {"match_id": match_id, "match_date": date, "competition": match.get("_competition", ""), "season": match.get("_season", ""), "home_team": home, "away_team": away, "player_id": p["player_id"], "player_name": p["player_name"], "team": p["team"], "position": p["position"], "minutes": min(mins, max_minute)}
        rec.update(p["stats"])
        rec["xg"] = p["xg"]
        rec["xa"] = p["xa"]
        result.append(rec)
    return result


def parse_time(value):
    if not value:
        return None
    try:
        parts = str(value).split(":")
        return float(parts[0]) + (float(parts[1]) / 60 if len(parts) > 1 else 0)
    except (ValueError, IndexError):
        return None


def event_minute(event):
    return float(event.get("minute", 0) or 0) + float(event.get("second", 0) or 0) / 60


def normalize_position(position):
    p = position.lower()
    if "goalkeeper" in p: return "Goalkeeper"
    if any(x in p for x in ("centre back", "center back", "full back", "wing back", "back")): return "Defender"
    if "defensive midfield" in p: return "Defensive Midfielder"
    if "midfield" in p or "wing" in p: return "Midfielder"
    if any(x in p for x in ("forward", "striker", "centre forward", "center forward")): return "Forward"
    return "Unknown"


METRICS = ["goals", "assists", "shots", "shots_on_target", "passes", "accurate_passes", "carries", "dribbles", "successful_dribbles", "pressures", "interceptions", "clearances", "recoveries", "tackles", "tackles_won", "blocks", "fouls", "miscontrols", "dispossessed", "saves"]


if __name__ == "__main__":
    main()
