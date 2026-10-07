"""Normalize Jolpica race collections without inventing historical availability."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pyarrow as pa

from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.time import require_utc

_UTC = pa.timestamp("us", tz="UTC")
EVENT_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("season", pa.int32(), nullable=False),
        pa.field("round", pa.int16(), nullable=False),
        pa.field("race_name", pa.string(), nullable=False),
        pa.field("circuit_id", pa.string(), nullable=False),
        pa.field("race_date", pa.date32(), nullable=False),
        pa.field("race_start_utc", _UTC),
        pa.field("available_at", _UTC),
    ]
)
ENTRY_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("constructor_id", pa.string()),
        pa.field("available_at", _UTC),
    ]
)
QUALIFYING_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("constructor_id", pa.string()),
        pa.field("position", pa.int16()),
        pa.field("q1_seconds", pa.float64()),
        pa.field("q2_seconds", pa.float64()),
        pa.field("q3_seconds", pa.float64()),
        pa.field("available_at", _UTC),
    ]
)
RESULT_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("constructor_id", pa.string()),
        pa.field("position", pa.int16()),
        pa.field("position_text", pa.string()),
        pa.field("grid", pa.int16()),
        pa.field("laps", pa.int16()),
        pa.field("points", pa.float64()),
        pa.field("status", pa.string()),
        pa.field("available_at", _UTC),
    ]
)
SPRINT_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("constructor_id", pa.string()),
        pa.field("number", pa.int16()),
        pa.field("position", pa.int16()),
        pa.field("position_text", pa.string()),
        pa.field("grid", pa.int16()),
        pa.field("laps", pa.int16()),
        pa.field("points", pa.float64()),
        pa.field("status", pa.string()),
        pa.field("available_at", _UTC),
    ]
)


def _required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _event_id(race: dict[str, Any]) -> EventId:
    try:
        season = _optional_int(race.get("season"), "season")
        round_number = _optional_int(race.get("round"), "round")
        if season is None or round_number is None:
            raise ValueError("missing season or round")
        return EventId(season, round_number)
    except ValueError as exc:
        raise ValueError("race has invalid season or round") from exc


def _entity_id(record: dict[str, Any], field: str, kind: EntityKind) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    nested = _required_mapping(value, field)
    key = {
        EntityKind.DRIVER: "driverId",
        EntityKind.CONSTRUCTOR: "constructorId",
        EntityKind.CIRCUIT: "circuitId",
    }[kind]
    return EntityId(kind, _required_string(nested.get(key), key)).value


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.isdecimal():
        raise ValueError(f"{name} must be a non-negative integer string")
    return int(value)


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not 0 <= parsed < float("inf"):
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _lap_seconds(value: Any, name: str) -> float | None:
    if value is None or value == "":
        return None
    text = _required_string(value, name)
    parts = text.split(":")
    if len(parts) != 2 or not parts[0].isdecimal():
        raise ValueError(f"{name} must be mm:ss.sss")
    seconds = _optional_float(parts[1], name)
    if seconds is None or seconds >= 60:
        raise ValueError(f"{name} seconds must be below 60")
    return int(parts[0]) * 60 + seconds


def normalize_schedule(races: list[dict[str, Any]], season: int) -> pa.Table:
    rows: list[dict[str, Any]] = []
    seen: set[EventId] = set()
    for raw_race in races:
        race = _required_mapping(raw_race, "race")
        event = _event_id(race)
        if event.season != season or event in seen:
            raise ValueError("schedule contains a wrong-season or duplicate event")
        seen.add(event)
        circuit_id = _entity_id(race, "Circuit", EntityKind.CIRCUIT)
        if circuit_id is None:
            raise ValueError("race is missing Circuit.circuitId")
        date_text = _required_string(race.get("date"), "date")
        try:
            race_date = date.fromisoformat(date_text)
            start = (
                datetime.fromisoformat(f"{date_text}T{race['time']}") if race.get("time") else None
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("race has invalid date or time") from exc
        if start is not None:
            require_utc(start, "race time")
        rows.append(
            {
                "event_id": event.partition(),
                "season": event.season,
                "round": event.round,
                "race_name": _required_string(race.get("raceName"), "raceName"),
                "circuit_id": circuit_id,
                "race_date": race_date,
                "race_start_utc": start,
                "available_at": None,
            }
        )
    return pa.Table.from_pylist(sorted(rows, key=lambda row: row["round"]), schema=EVENT_SCHEMA)


def _session_rows(races: list[dict[str, Any]], event: EventId, field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_race in races:
        race = _required_mapping(raw_race, "race")
        if _event_id(race) != event:
            raise ValueError("session response contains a different event")
        raw_results = race.get(field, [])
        if not isinstance(raw_results, list):
            raise ValueError(f"{field} must be a list")
        for raw_result in raw_results:
            result = _required_mapping(raw_result, field)
            driver_id = _entity_id(result, "Driver", EntityKind.DRIVER)
            if driver_id is None or driver_id in seen:
                raise ValueError(f"{field} has a missing or duplicate driver")
            seen.add(driver_id)
            rows.append(
                {
                    "event_id": event.partition(),
                    "driver_id": driver_id,
                    "constructor_id": _entity_id(result, "Constructor", EntityKind.CONSTRUCTOR),
                    "available_at": None,
                    "source": result,
                }
            )
    return rows


def normalize_qualifying(races: list[dict[str, Any]], event: EventId) -> pa.Table:
    rows = _session_rows(races, event, "QualifyingResults")
    for row in rows:
        source = row.pop("source")
        row["position"] = _optional_int(source.get("position"), "position")
        for session in ("Q1", "Q2", "Q3"):
            row[f"{session.lower()}_seconds"] = _lap_seconds(source.get(session), session)
    return pa.Table.from_pylist(rows, schema=QUALIFYING_SCHEMA)


def normalize_results(races: list[dict[str, Any]], event: EventId) -> pa.Table:
    rows = _session_rows(races, event, "Results")
    for row in rows:
        source = row.pop("source")
        row["position"] = _optional_int(source.get("position"), "position")
        row["position_text"] = source.get("positionText")
        row["grid"] = _optional_int(source.get("grid"), "grid")
        row["laps"] = _optional_int(source.get("laps"), "laps")
        row["points"] = _optional_float(source.get("points"), "points")
        row["status"] = source.get("status")
    return pa.Table.from_pylist(rows, schema=RESULT_SCHEMA)


def normalize_sprint(races: list[dict[str, Any]], event: EventId) -> pa.Table:
    """Sprint classification; ``grid`` is the sprint starting grid, not the race grid."""
    rows = _session_rows(races, event, "SprintResults")
    for row in rows:
        source = row.pop("source")
        row["number"] = _optional_int(source.get("number"), "number")
        row["position"] = _optional_int(source.get("position"), "position")
        row["position_text"] = source.get("positionText")
        row["grid"] = _optional_int(source.get("grid"), "grid")
        row["laps"] = _optional_int(source.get("laps"), "laps")
        row["points"] = _optional_float(source.get("points"), "points")
        row["status"] = source.get("status")
    return pa.Table.from_pylist(rows, schema=SPRINT_SCHEMA)


def normalize_entries(qualifying: pa.Table, results: pa.Table) -> pa.Table:
    entries: dict[str, dict[str, Any]] = {}
    for table in (qualifying, results):
        for row in table.select(["event_id", "driver_id", "constructor_id"]).to_pylist():
            driver_id = row["driver_id"]
            previous = entries.get(driver_id)
            if previous is not None:
                if previous["event_id"] != row["event_id"]:
                    raise ValueError("entries contain different events")
                if (
                    previous["constructor_id"]
                    and row["constructor_id"]
                    and previous["constructor_id"] != row["constructor_id"]
                ):
                    raise ValueError("driver has conflicting constructors")
                previous["constructor_id"] = previous["constructor_id"] or row["constructor_id"]
            else:
                entries[driver_id] = {**row, "available_at": None}
    return pa.Table.from_pylist([entries[key] for key in sorted(entries)], schema=ENTRY_SCHEMA)
