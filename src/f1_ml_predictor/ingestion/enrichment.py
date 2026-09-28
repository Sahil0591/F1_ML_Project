"""Persist bounded session and forecast snapshots with known capture times."""

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.ingestion.cache import RawCache
from f1_ml_predictor.ingestion.parquet import write_partition
from f1_ml_predictor.normalization.sessions import normalize_summaries, summarize_openf1
from f1_ml_predictor.normalization.weather import normalize_forecast
from f1_ml_predictor.paths import StoragePaths
from f1_ml_predictor.sources.fastf1 import load_session_summary
from f1_ml_predictor.sources.open_meteo import ForecastSnapshot
from f1_ml_predictor.sources.openf1 import OpenF1Client
from f1_ml_predictor.time import require_known_by, require_utc


@dataclass(frozen=True, slots=True)
class EnrichmentReport:
    rows: int
    path: Path
    written: bool
    unmapped_laps: int = 0


def event_race(paths: StoragePaths, event: EventId) -> dict[str, Any]:
    schedule = RawCache(paths.raw / "jolpica").load(f"season={event.season}/schedule")
    if schedule is None:
        raise ValueError("event schedule is required for source mapping")
    races = [race for race in schedule.items if int(race["round"]) == event.round]
    if len(races) != 1:
        raise ValueError("event is absent or duplicated in schedule")
    return races[0]


def driver_crosswalk(paths: StoragePaths, event: EventId) -> dict[str, str]:
    cache = RawCache(paths.raw / "jolpica")
    aliases: dict[str, str] = {}
    for name, field in (("qualifying", "QualifyingResults"), ("results", "Results")):
        record = cache.load(f"{event.partition()}/{name}")
        if record is None:
            continue
        for race in record.items:
            for entry in race.get(field, []):
                number = entry.get("number")
                if number is None:
                    continue
                number = str(number)
                if not number.isdecimal() or int(number) < 1:
                    raise ValueError("invalid driver number in source crosswalk")
                driver_id = EntityId(EntityKind.DRIVER, entry["Driver"]["driverId"]).value
                if number in aliases and aliases[number] != driver_id:
                    raise ValueError("conflicting driver number crosswalk")
                aliases[number] = driver_id
    if not aliases:
        raise ValueError("ingest event qualifying with driver numbers before session enrichment")
    return aliases


def ingest_openf1_session(
    paths: StoragePaths,
    event: EventId,
    session_key: int,
    client: OpenF1Client,
    *,
    refresh: bool = False,
) -> EnrichmentReport:
    if event.season < 2023:
        raise ValueError("OpenF1 historical coverage starts in 2023")
    aliases = driver_crosswalk(paths, event)
    jolpica_cache = RawCache(paths.raw / "jolpica")
    mapping_records = [
        jolpica_cache.load(f"season={event.season}/schedule"),
        jolpica_cache.load(f"{event.partition()}/qualifying"),
        jolpica_cache.load(f"{event.partition()}/results"),
    ]
    cache = RawCache(paths.raw / "openf1")
    records = {}
    for endpoint in ("sessions", "laps", "stints", "pit"):
        key = f"{event.partition()}/session={session_key}/{endpoint}"
        cached = cache.load(key)
        if cached is None or refresh:
            payload = client.collection(endpoint, session_key)
            cached = cache.save(
                key, payload, datetime.now(UTC), f"/v1/{endpoint}?session_key={session_key}"
            )
        records[endpoint] = cached
        if any(item.get("session_key") != session_key for item in cached.items):
            raise ValueError("OpenF1 records do not belong to the selected session")
    sessions = records["sessions"].items
    if len(sessions) != 1 or sessions[0].get("session_key") != session_key:
        raise ValueError("OpenF1 session identity is ambiguous")
    session = sessions[0]
    race = event_race(paths, event)
    end = datetime.fromisoformat(session["date_end"])
    require_utc(end, "session end")
    day_gap = (date.fromisoformat(race["date"]) - end.date()).days
    if not 0 <= day_gap <= 4 or end.year != event.season:
        raise ValueError("OpenF1 session date does not match the selected event")
    all_laps = records["laps"].items
    laps = [lap for lap in all_laps if str(lap.get("driver_number")) in aliases]
    captured_at = max(
        record.retrieved_at
        for record in [*records.values(), *mapping_records]
        if record is not None
    )
    rows = summarize_openf1(
        event, session, laps, records["stints"].items, records["pit"].items, aliases, captured_at
    )
    table = normalize_summaries(rows, "openf1")
    digest = hashlib.sha256(
        json.dumps(
            {
                "inputs": [
                    record.sha256
                    for record in [*records.values(), *mapping_records]
                    if record is not None
                ],
                "aliases": aliases,
            },
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    output = (
        paths.normalized
        / "session_summaries"
        / event.partition()
        / f"openf1-{session_key}-{digest}-v2.parquet"
    )
    written = write_partition(table, output, digest, "2")
    return EnrichmentReport(table.num_rows, output, written, len(all_laps) - len(laps))


def ingest_fastf1_session(paths: StoragePaths, event: EventId, code: str) -> EnrichmentReport:
    rows = load_session_summary(event, code, driver_crosswalk(paths, event), paths.raw / "fastf1")
    capture = rows[0]["available_at"] if rows else datetime.now(UTC)
    values = [{key: value for key, value in row.items() if key != "available_at"} for row in rows]
    record = RawCache(paths.raw / "fastf1_summaries").save(
        f"{event.partition()}/{code}",
        values,
        capture,
        f"fastf1/{event.season}/{event.round}/{code}",
    )
    table = normalize_summaries(
        [{**row, "available_at": record.retrieved_at} for row in record.items], "fastf1"
    )
    output = (
        paths.normalized
        / "session_summaries"
        / event.partition()
        / f"fastf1-{code}-{record.sha256}-v2.parquet"
    )
    return EnrichmentReport(
        table.num_rows, output, write_partition(table, output, record.sha256, "2")
    )


def persist_forecast(
    paths: StoragePaths, event: EventId, snapshot: ForecastSnapshot
) -> EnrichmentReport:
    normalize_forecast(event, snapshot)
    payload: dict[str, Any] = {
        "response": snapshot.payload,
        "run_initialized_at": snapshot.run_initialized_at.isoformat()
        if snapshot.run_initialized_at
        else None,
        "known_available_at": snapshot.available_at.isoformat()
        if snapshot.run_initialized_at
        else None,
        "availability_evidence": snapshot.availability_evidence,
    }
    record = RawCache(paths.raw / "open_meteo").save(
        f"{event.partition()}/forecast", [payload], snapshot.captured_at, snapshot.request_path
    )
    available_at = snapshot.available_at if snapshot.run_initialized_at else record.retrieved_at
    stored = replace(snapshot, captured_at=record.retrieved_at, available_at=available_at)
    table = normalize_forecast(event, stored)
    output = paths.normalized / "forecasts" / event.partition() / f"{record.sha256}-v2.parquet"
    return EnrichmentReport(
        table.num_rows, output, write_partition(table, output, record.sha256, "2")
    )


def record_fia_evidence(
    paths: StoragePaths,
    event: EventId,
    document_url: str,
    document_kind: str,
    published_at: datetime,
) -> None:
    require_known_by(published_at, datetime.now(UTC))
    url = urlparse(document_url)
    host = url.hostname or ""
    if url.scheme != "https" or not (host == "fia.com" or host.endswith(".fia.com")):
        raise ValueError("FIA evidence must link to an official HTTPS document")
    if document_kind not in {"grid", "classification", "decision", "regulations"}:
        raise ValueError("unsupported FIA document kind")
    RawCache(paths.raw / "fia").save(
        f"{event.partition()}/{document_kind}",
        [{"document_url": document_url, "published_at": published_at.isoformat()}],
        datetime.now(UTC),
        document_url,
    )
