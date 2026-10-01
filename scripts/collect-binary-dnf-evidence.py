"""Retain raw OpenF1 and Jolpica race outcome evidence for Gold audits.

All API responses are current observations. The later audit must compare them
with the already retained exact FIA final classification before assigning labels.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/dnf_audit_v1"
REGISTRY = ROOT / "data/benchmarks/gold_core_registry.json"
OPENF1 = "https://api.openf1.org/v1"
JOLPICA = "https://api.jolpi.ca/ergast/f1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def retain(client: httpx.Client, url: str, *, pace_seconds: float = 0) -> dict[str, Any]:
    request_dir = RAW / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    path = request_dir / f"{key}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        object_path = ROOT / record["path"]
        if hashlib.sha256(object_path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"retained DNF response changed: {url}")
        return record
    if pace_seconds:
        time.sleep(pace_seconds)
    response = client.get(url)
    response.raise_for_status()
    if str(response.url) != url:
        raise ValueError(f"DNF source redirected: {url}")
    payload = response.content
    parsed = json.loads(payload)
    if not isinstance(parsed, dict | list):
        raise ValueError("DNF response is not a JSON object or array")
    digest = hashlib.sha256(payload).hexdigest()
    object_path = RAW / "objects" / f"{digest}.json"
    object_path.parent.mkdir(parents=True, exist_ok=True)
    if object_path.exists() and object_path.read_bytes() != payload:
        raise ValueError("DNF content address collision")
    object_path.write_bytes(payload)
    record = {
        "url": url,
        "path": object_path.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "content_type": response.headers.get("content-type"),
        "last_modified": response.headers.get("last-modified"),
        "classification": "current_state_only",
    }
    path.write_bytes(canonical(record))
    return record


def payload(record: dict[str, Any]) -> Any:
    source = ROOT / record["path"]
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise ValueError("DNF raw source hash mismatch")
    return json.loads(content)


def main() -> None:
    registry_bytes = REGISTRY.read_bytes()
    races = json.loads(registry_bytes)["races"]
    result: dict[str, Any] = {
        "version": 1,
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "source_status": "current_state_only_cross_checks",
        "races": [],
    }
    with httpx.Client(
        timeout=30, follow_redirects=False, headers={"User-Agent": "f1-ml-predictor/0.1.0"}
    ) as client:
        sessions: dict[int, dict[str, Any] | None] = {}
        for year in sorted({int(race["event_id"][7:11]) for race in races}):
            try:
                sessions[year] = retain(
                    client, f"{OPENF1}/sessions?year={year}&session_name=Race", pace_seconds=2.1
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
                sessions[year] = None
        for race in races:
            event = race["event_id"]
            season = int(event[7:11])
            round_number = int(event[-2:])
            entry: dict[str, Any] = {"event_id": event, "status": "unresolved"}
            try:
                jolpica = retain(
                    client,
                    f"{JOLPICA}/{season}/{round_number}/results/?limit=100",
                    pace_seconds=0.3,
                )
                source_races = payload(jolpica)["MRData"]["RaceTable"]["Races"]
                if len(source_races) != 1:
                    raise ValueError("Jolpica race result is missing or ambiguous")
                source_race = source_races[0]
                if (int(source_race["season"]), int(source_race["round"])) != (
                    season,
                    round_number,
                ):
                    raise ValueError("Jolpica result event identity differs")
                season_sessions = sessions[season]
                if season_sessions is None:
                    raise ValueError("OpenF1 has no race sessions for this season")
                start = datetime.fromisoformat(
                    source_race["date"] + "T" + source_race["time"].replace("Z", "+00:00")
                )
                matches = [
                    session
                    for session in payload(season_sessions)
                    if session["session_name"] == "Race"
                    and session["year"] == season
                    and not session.get("is_cancelled", False)
                    and abs((datetime.fromisoformat(session["date_start"]) - start).total_seconds())
                    < 8 * 3600
                ]
                if len(matches) != 1:
                    raise ValueError("OpenF1 race session is missing or ambiguous")
                session = matches[0]
                openf1 = retain(
                    client,
                    f"{OPENF1}/session_result?session_key={session['session_key']}",
                    pace_seconds=2.1,
                )
                entry.update(
                    {
                        "status": "captured",
                        "jolpica": jolpica,
                        "openf1_sessions": season_sessions,
                        "openf1_session_key": session["session_key"],
                        "openf1_session_result": openf1,
                        "jolpica_race_start_utc": start.isoformat(),
                    }
                )
            except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
                entry["reason"] = str(exc)
            result["races"].append(entry)
            print(f"{event}: {entry['status']}", flush=True)
    report = RAW / f"capture-{hashlib.sha256(canonical(result)).hexdigest()}.json"
    report.write_bytes(canonical(result))
    print(f"report: {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
