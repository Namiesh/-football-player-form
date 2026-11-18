"""Incrementally build a Premier League player-match dataset from the supplied S3 scrape.

Uses the local AWS credential chain (AWS_PROFILE or the default profile). No
football-data API is called. Only matchstats.json objects are downloaded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import boto3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUCKET = os.getenv("OPTA_S3_BUCKET", "scrape-opta-data")
PREFIX = os.getenv("OPTA_S3_PREFIX", "football")
REGION = os.getenv("AWS_REGION", "ap-south-1")
RAW = ROOT / "data" / "raw" / "opta_s3" / "matchstats"
OUT = ROOT / "data" / "processed" / "player_match_stats_s3.csv"
CURRENT_OUT = ROOT / "data" / "processed" / "player_match_stats_2026_2027.csv"

STAT_ALIASES = {
    "minsPlayed": "minutes", "goals": "goals", "goalAssist": "assists",
    "totalScoringAtt": "shots", "ontargetScoringAtt": "shots_on_target",
    "shotOffTarget": "shots_off_target", "blockedScoringAtt": "blocks",
    "totalPass": "passes", "accuratePass": "accurate_passes",
    "totalTackle": "tackles", "wonTackle": "tackles_won",
    "totalClearance": "clearances", "interceptionWon": "interceptions",
    "ballRecovery": "recoveries", "duelWon": "duels_won",
    "wonContest": "duels_won", "aerialWon": "aerials_won",
    "fouls": "fouls", "saves": "saves", "goalsConceded": "goals_conceded",
    "yellowCard": "yellow_cards", "redCard": "red_cards",
    "cleanSheet": "clean_sheet", "totalOffside": "offsides",
    "totalCross": "crosses", "accurateCross": "accurate_crosses",
    "wasFouled": "was_fouled", "formationPlace": "formation_place",
}
NUMERIC_FIELDS = sorted(set(STAT_ALIASES.values()))


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _position_group(value: str) -> str:
    value = str(value or "").strip().casefold()
    if value in {"g", "gk", "goalkeeper", "keeper"}:
        return "Goalkeeper"
    if value in {"dm", "am", "cm", "lm", "rm", "midfielder", "midfield",
                 "attacking midfielder", "defensive midfielder"}:
        return "Midfielder"
    if value in {"striker", "forward", "attacker"}:
        return "Forward"
    if value in {"wing back", "wing-back"}:
        return "Defender"
    if value in {"d", "dc", "dl", "dr", "defender", "defence"}:
        return "Defender"
    if value in {"m", "mc", "ml", "mr", "midfielder", "midfield"}:
        return "Midfielder"
    if value in {"f", "fw", "st", "forward", "attacker", "striker"}:
        return "Forward"
    # Opta lineup position values are usually role names, but some feeds use
    # more specific codes such as DM, AM, WB, or CB.
    if value.startswith(("gk", "goalkeeper")):
        return "Goalkeeper"
    if value.startswith(("m", "dm", "am", "mid")):
        return "Midfielder"
    if value.startswith(("d", "cb", "wb", "def")):
        return "Defender"
    if value.startswith(("f", "fw", "st", "att")):
        return "Forward"
    return "Unknown"


def _match_metadata(document: dict, calendar_year: int) -> dict:
    info = document.get("matchInfo") or {}
    live = document.get("liveData") or {}
    contestants = info.get("contestant") or []
    teams = {}
    team_ids = {}
    for item in contestants:
        side = str(item.get("position") or "").casefold()
        cid = str(item.get("id") or item.get("contestantId") or "")
        name = item.get("name") or item.get("shortName") or cid
        if side:
            teams[side] = name
        if cid:
            team_ids[cid] = name
    home = teams.get("home", "")
    away = teams.get("away", "")
    calendar = info.get("tournamentCalendar") or {}
    season = str(calendar.get("name") or "")
    match = re.search(r"(20\d{2})\s*[/\-]\s*(20\d{2})", season)
    if match:
        season = f"{match.group(1)}/{match.group(2)}"
    else:
        season = f"{calendar_year}/{calendar_year + 1}" if calendar_year >= 7 else f"{calendar_year - 1}/{calendar_year}"
    comp = info.get("competition") or {}
    match_date = info.get("date") or info.get("localDate")
    match_time = info.get("time") or info.get("localTime")
    if match_date and "T" in str(match_date):
        date_value = str(match_date)
    elif match_date and match_time:
        date_value = f"{str(match_date).rstrip('Z')}T{str(match_time).rstrip('Z')}+00:00"
    else:
        date_value = match_date
    scores = ((live.get("matchDetails") or {}).get("scores") or {})
    ft = scores.get("ft") or scores.get("total") or {}
    home_goals = _number(ft.get("home"))
    away_goals = _number(ft.get("away"))
    return {
        "match_id": info.get("id") or info.get("matchId"),
        "match_date": pd.to_datetime(date_value, errors="coerce", utc=True),
        "competition": comp.get("name") or "Premier League",
        "season": season,
        "home_team": home,
        "away_team": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "match_status": (live.get("matchDetails") or {}).get("matchStatus", ""),
        "team_ids": team_ids,
    }


def parse_match(document: dict, calendar_year: int) -> list[dict]:
    meta = _match_metadata(document, calendar_year)
    if not meta["match_id"] or pd.isna(meta["match_date"]):
        return []
    if meta["home_goals"] is None or meta["away_goals"] is None:
        return []  # Exclude fixtures without a settled score.
    rows = []
    for lineup in (document.get("liveData") or {}).get("lineUp") or []:
        team_id = str(lineup.get("contestantId") or "")
        team = meta["team_ids"].get(team_id, "")
        side = str(lineup.get("contestantId") or "")
        is_home = team == meta["home_team"] or side in {
            str(c.get("id")) for c in ((document.get("matchInfo") or {}).get("contestant") or [])
            if str(c.get("position", "")).casefold() == "home"
        }
        opponent = meta["away_team"] if is_home else meta["home_team"]
        for player in lineup.get("player") or []:
            stats = {}
            for item in player.get("stat") or []:
                dest = STAT_ALIASES.get(item.get("type"))
                value = _number(item.get("value"))
                if dest and value is not None:
                    stats[dest] = value
            minutes = stats.get("minutes", 0.0)
            if minutes <= 0:
                continue
            player_id = player.get("playerId") or player.get("id")
            if player_id is None:
                continue
            name = player.get("matchName") or " ".join(
                x for x in (player.get("firstName"), player.get("lastName")) if x
            )
            row = {
                "match_id": str(meta["match_id"]), "match_date": meta["match_date"],
                "competition": meta["competition"], "season": meta["season"],
                "home_team": meta["home_team"], "away_team": meta["away_team"],
                "team": team, "opponent": opponent, "player_id": str(player_id),
                "player_name": name, "position_id": player.get("position"),
                "position": _position_group(player.get("position")),
                "minutes": minutes, "is_home": int(is_home),
                "home_goals": meta["home_goals"], "away_goals": meta["away_goals"],
                "goals_for": meta["home_goals"] if is_home else meta["away_goals"],
                "goals_against": meta["away_goals"] if is_home else meta["home_goals"],
                "xg": None, "xa": None,
            }
            row.update({field: stats.get(field) for field in NUMERIC_FIELDS if field not in row})
            passes = row.get("passes") or 0
            accurate = row.get("accurate_passes") or 0
            row["pass_accuracy"] = accurate / passes if passes > 0 else None
            rows.append(row)
    return rows


def _cache_path(key: str) -> Path:
    match = re.search(r"football/(\d+)/premier-league-england/(\d{2})/matches/([^/]+)/matchstats\.json$", key)
    if not match:
        raise ValueError(f"Unexpected S3 matchstats key layout: {key}")
    base = RAW / "continuous" if key.startswith("continuous-scrape-data/") else RAW
    return base / match.group(1) / match.group(2) / f"{match.group(3)}.json"


def load_s3_documents(years: list[int], bucket: str, prefix: str) -> list[tuple[int, dict]]:
    session = boto3.Session()
    client = session.client("s3", region_name=REGION)
    paginator = client.get_paginator("list_objects_v2")
    documents = []
    for year in years:
        historical_prefix = f"{prefix.rstrip('/')}/{year}/premier-league-england/"
        continuous_prefix = f"continuous-scrape-data/{prefix.rstrip('/')}/{year}/premier-league-england/"
        objects = []
        for source, s3_prefix in (("historical", historical_prefix), ("continuous", continuous_prefix)):
            source_objects = []
            for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
                source_objects.extend(obj for obj in page.get("Contents", [])
                                      if obj["Key"].endswith("/matchstats.json"))
            objects.extend(source_objects)
            print(f"{year} {source}: found {len(source_objects)} matchstats objects")
        for index, obj in enumerate(objects, start=1):
            path = _cache_path(obj["Key"])
            etag_path = path.with_suffix(".etag")
            etag = str(obj.get("ETag", "")).strip('"')
            cached_etag = etag_path.read_text(encoding="utf-8").strip() if etag_path.exists() else ""
            if not path.exists() or cached_etag != etag:
                path.parent.mkdir(parents=True, exist_ok=True)
                response = client.get_object(Bucket=bucket, Key=obj["Key"])
                content = response["Body"].read()
                path.write_bytes(content)
                etag_path.write_text(etag, encoding="utf-8")
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                print(f"Skipping invalid JSON {obj['Key']}: {exc}")
                continue
            documents.append((year, document))
            if index % 100 == 0 or index == len(objects):
                print(f"{year}: read {index}/{len(objects)} cached matchstats files")
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="2023,2024,2025,2026,2027",
                        help="Comma-separated calendar year folders under the Premier League prefix.")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args()
    years = [int(value.strip()) for value in args.years.split(",") if value.strip()]
    docs = load_s3_documents(years, args.bucket, args.prefix)
    rows = [row for year, doc in docs for row in parse_match(doc, year)]
    if not rows:
        raise SystemExit("No finished player appearances found. Check the bucket, prefix, and local AWS profile.")
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["match_id", "player_id"], keep="last")
    df = df.sort_values(["player_id", "match_date", "match_id"]).reset_index(drop=True)
    # Opta marks an in-game substitute's role as "Substitute". Fill that label
    # from the player's most common actual role in the same season, when known.
    known = df[df["position"] != "Unknown"]
    season_role = (known.groupby(["player_id", "season"])["position"]
                   .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "Unknown"))
    inferred = pd.MultiIndex.from_frame(df[["player_id", "season"]]).map(season_role)
    df.loc[df["position"] == "Unknown", "position"] = inferred[df["position"].eq("Unknown")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    current = df[df["season"].astype(str).str.contains(r"2026\s*/\s*2027", regex=True)]
    current.to_csv(CURRENT_OUT, index=False)
    print(f"Saved {len(df):,} player appearances across {df.match_id.nunique():,} finished matches and {df.player_id.nunique():,} stable player IDs to {OUT}")
    print(f"Saved {len(current):,} 2026/27 appearances to {CURRENT_OUT}")
    print("Note: the supplied S3 matchstats sample has no xG or injury-status fields; these are left missing, not fabricated.")


if __name__ == "__main__":
    main()
