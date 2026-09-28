"""Build pre-race features without consulting outcome data from the target event."""

import hashlib
import json
import math
from datetime import datetime, timedelta
from statistics import mean
from typing import Any

import pyarrow as pa

from f1_ml_predictor.features.contracts import FeatureInputs, PublishedTable, ResultVersion
from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.time import require_known_by, require_utc

FEATURE_VERSION = "1"
NUMERIC_FEATURES = (
    "qualifying_position",
    "qualifying_last_session_seconds",
    "grid_position",
    "teammate_qualifying_position_delta",
    "recent_finish_mean",
    "recent_dnf_rate",
    "constructor_recent_finish_mean",
    "recent_pit_stop_seconds",
    "practice_best_seconds",
    "practice_median_seconds",
    "tyre_age_mean",
    "tyre_compound_count",
    "forecast_temperature_2m",
    "forecast_precipitation_probability",
    "forecast_wind_speed_10m",
    "driver_championship_points",
    "constructor_championship_points",
    "circuit_length_km",
    "is_street_circuit",
)
FEATURE_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("constructor_id", pa.string(), nullable=False),
        pa.field("circuit_id", pa.string(), nullable=False),
        pa.field("prediction_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("feature_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("feature_version", pa.string(), nullable=False),
        pa.field("history_count", pa.int32(), nullable=False),
        pa.field("form_window", pa.int32(), nullable=False),
        pa.field("session_source", pa.string()),
        pa.field("provenance", pa.string(), nullable=False),
        *[pa.field(name, pa.float64()) for name in NUMERIC_FEATURES],
        *[pa.field(f"{name}_missing", pa.bool_(), nullable=False) for name in NUMERIC_FEATURES],
    ]
)


def _known(value: PublishedTable | None, cutoff: datetime) -> bool:
    return value is not None and value.available_at is not None and value.available_at <= cutoff


def _latest(versions: tuple[PublishedTable, ...], cutoff: datetime) -> PublishedTable | None:
    known = [value for value in versions if _known(value, cutoff)]
    if not known:
        return None
    newest = max(value.available_at for value in known if value.available_at is not None)
    selected = [value for value in known if value.available_at == newest]
    if len(selected) != 1:
        raise ValueError("ambiguous publication versions at the same timestamp")
    return selected[0]


def _rows(publication: PublishedTable, event_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = publication.table.to_pylist()
    if any(row.get("event_id") != event_id for row in rows):
        raise ValueError("publication contains a different event")
    return rows


def _by_driver(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        driver = EntityId(EntityKind.DRIVER, row["driver_id"]).value
        if driver in result:
            raise ValueError("publication has duplicate drivers")
        result[driver] = row
    return result


def _number(value: Any, name: str, *, positive: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data or missing")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be nonnegative or missing")
    return float(value)


def _average(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _history(inputs: FeatureInputs, cutoff: datetime) -> list[ResultVersion]:
    groups: dict[EventId, list[ResultVersion]] = {}
    for version in inputs.history:
        if (
            version.event < inputs.event.event
            and version.race_completed_at < cutoff
            and _known(version.publication, cutoff)
        ):
            groups.setdefault(version.event, []).append(version)
    result = []
    for event, versions in sorted(groups.items()):
        publication = _latest(tuple(version.publication for version in versions), cutoff)
        selected = next(version for version in versions if version.publication is publication)
        _by_driver(_rows(selected.publication, event.partition()))
        result.append(selected)
    return result


def build_snapshot(
    inputs: FeatureInputs,
    prediction_timestamp: datetime,
    *,
    form_window: int = 5,
    session_source: str = "fastf1",
) -> pa.Table:
    require_utc(prediction_timestamp, "prediction_timestamp")
    spec = inputs.event
    require_known_by(spec.available_at, prediction_timestamp)
    if not spec.qualifying_completed_at <= prediction_timestamp < spec.race_start:
        raise ValueError("cutoff must be after qualifying and before race start")
    if isinstance(form_window, bool) or not isinstance(form_window, int) or form_window < 1:
        raise ValueError("form_window must be a positive integer")
    if session_source not in {"fastf1", "openf1"}:
        raise ValueError("session_source must be fastf1 or openf1")
    roster = _latest(inputs.rosters, prediction_timestamp)
    qualifying = _latest(inputs.qualifying, prediction_timestamp)
    if roster is None or qualifying is None:
        raise ValueError("published roster and qualifying are required by the cutoff")
    assert qualifying.available_at is not None
    require_known_by(spec.qualifying_completed_at, qualifying.available_at)
    event_id = spec.event.partition()
    entrants = _by_driver(_rows(roster, event_id))
    if not entrants:
        raise ValueError("roster must contain drivers")
    q_rows = _by_driver(_rows(qualifying, event_id))
    if not q_rows:
        raise ValueError("qualifying publication must contain drivers")
    if set(q_rows) - set(entrants):
        raise ValueError("qualifying contains a driver absent from the roster")
    used = [roster, qualifying]
    history = _history(inputs, prediction_timestamp)
    historical_rows = [
        (version, _rows(version.publication, version.event.partition())) for version in history
    ]
    used.extend(version.publication for version in history)

    # Select one source per session to avoid averaging incompatible quality filters.
    sessions: dict[str, PublishedTable] = {}
    for publication in inputs.sessions:
        if not _known(publication, prediction_timestamp):
            continue
        records = _rows(publication, event_id)
        codes = {row.get("session_code") for row in records}
        if not codes <= {"FP1", "FP2", "FP3", "Q"}:
            raise ValueError("race sessions cannot enter pre-race features")
        sources = {row.get("source") for row in records}
        if sources != {session_source} or not records:
            continue
        if len(codes) != 1:
            raise ValueError("session publication must contain one session")
        code = next(iter(codes))
        assert isinstance(code, str) and publication.available_at is not None
        previous = sessions.get(code)
        if previous is None:
            sessions[code] = publication
            continue
        assert previous.available_at is not None
        if publication.available_at > previous.available_at:
            sessions[code] = publication
        elif publication.available_at == previous.available_at:
            raise ValueError("ambiguous session versions")
    used.extend(sessions.values())
    session_rows = [
        row
        for publication in sessions.values()
        for row in _rows(publication, event_id)
        if row["session_code"] != "Q"
    ]

    weather = _latest(inputs.forecasts, prediction_timestamp)
    weather_row = None
    if weather is not None:
        candidates = []
        for row in _rows(weather, event_id):
            if "valid_at" not in row or "captured_at" not in row:
                raise ValueError("weather input must be a forecast, not observed weather")
            require_utc(row["valid_at"], "valid_at")
            require_utc(row["captured_at"], "captured_at")
            available = row.get("available_at")
            if available is None:
                raise ValueError("forecast availability is required")
            require_known_by(available, prediction_timestamp)
            require_known_by(available, row["captured_at"])
            if not row.get("run_initialized_at") and available != row["captured_at"]:
                raise ValueError("live forecast availability cannot be backdated")
            if row.get("run_initialized_at"):
                require_known_by(row["run_initialized_at"], available)
                evidence = row.get("availability_evidence")
                if not isinstance(evidence, str) or not evidence.strip():
                    raise ValueError("archived forecast release evidence is required")
            if spec.race_start <= row["valid_at"] < spec.race_start + timedelta(hours=1):
                candidates.append(row)
        if len(candidates) > 1:
            raise ValueError("ambiguous race-start forecast targets")
        if candidates:
            weather_row = candidates[0]
            used.append(weather)

    standings_rows: dict[str, dict[str, Any]] = {}
    if _known(inputs.standings, prediction_timestamp):
        assert inputs.standings is not None
        rows = inputs.standings.table.to_pylist()
        if any(
            row.get("season") != spec.event.season
            or not isinstance(row.get("round"), int)
            or not 0 <= row["round"] < spec.event.round
            for row in rows
        ):
            raise ValueError("standings must precede the target round in the same season")
        if len({row["round"] for row in rows}) > 1:
            raise ValueError("standings publication must contain one common round")
        standings_rows = _by_driver(rows)
        used.append(inputs.standings)
    circuit_row = None
    if _known(inputs.circuit, prediction_timestamp):
        assert inputs.circuit is not None
        rows = inputs.circuit.table.to_pylist()
        if len(rows) != 1 or rows[0].get("circuit_id") != spec.circuit_id:
            raise ValueError("circuit metadata does not match the selected circuit")
        circuit_row = rows[0]
        used.append(inputs.circuit)
    feature_timestamp = max(
        [
            spec.available_at,
            *[value.available_at for value in used if value.available_at is not None],
        ]
    )
    records = []
    for publication in used:
        canonical_rows = sorted(
            json.dumps(row, sort_keys=True, default=lambda item: item.isoformat(), allow_nan=False)
            for row in publication.table.to_pylist()
        )
        records.append(
            {
                "reference": publication.evidence_reference,
                "available_at": publication.available_at.isoformat()
                if publication.available_at
                else None,
                "sha256": hashlib.sha256(json.dumps(canonical_rows).encode("utf-8")).hexdigest(),
            }
        )
    provenance = json.dumps(
        {
            "event_reference": spec.evidence_reference,
            "event_available_at": spec.available_at.isoformat(),
            "event_id": spec.event.partition(),
            "circuit_id": spec.circuit_id,
            "race_start": spec.race_start.isoformat(),
            "qualifying_completed_at": spec.qualifying_completed_at.isoformat(),
            "history": [
                {
                    "event_id": version.event.partition(),
                    "race_completed_at": version.race_completed_at.isoformat(),
                    "reference": version.publication.evidence_reference,
                }
                for version in history
            ],
            "inputs": sorted(records, key=lambda record: json.dumps(record, sort_keys=True)),
            "session_source_policy": session_source,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    output = []
    for driver, entrant in sorted(entrants.items()):
        constructor = EntityId(EntityKind.CONSTRUCTOR, entrant["constructor_id"]).value
        q = q_rows.get(driver, {})
        q_times = [
            _number(q.get(name), name, positive=True)
            for name in ("q3_seconds", "q2_seconds", "q1_seconds")
        ]
        if q.get("constructor_id") not in {None, constructor}:
            raise ValueError("qualifying and roster constructor identities conflict")
        driver_history = [
            row for _, rows in historical_rows for row in rows if row["driver_id"] == driver
        ][-form_window:]
        constructor_history = [
            row
            for _, rows in historical_rows[-form_window:]
            for row in rows
            if row.get("constructor_id") == constructor
        ]
        dnfs = [row.get("dnf") for row in driver_history if row.get("dnf") is not None]
        if any(not isinstance(value, bool) for value in dnfs):
            raise ValueError("DNF labels must be explicit audited booleans or missing")
        dnf_values = [value for value in dnfs if isinstance(value, bool)]
        q_position = _number(q.get("position"), "qualifying position", positive=True)
        teammates = [
            _number(row.get("position"), "teammate position", positive=True)
            for other, row in q_rows.items()
            if other != driver and entrants[other]["constructor_id"] == constructor
        ]
        teammate_position = _average(teammates)
        practice = [row for row in session_rows if row["driver_id"] == driver]
        compounds = {value["compound"] for value in practice if value.get("compound")}
        practice_best = [
            _number(row.get("best_lap_seconds"), "practice best", positive=True) for row in practice
        ]
        best_present = [value for value in practice_best if value is not None]
        standings = standings_rows.get(driver, {})
        if standings.get("constructor_id") not in {None, constructor}:
            raise ValueError("standings constructor identity conflicts with the roster")
        row = {
            "event_id": event_id,
            "driver_id": driver,
            "constructor_id": constructor,
            "circuit_id": spec.circuit_id,
            "prediction_timestamp": prediction_timestamp,
            "feature_timestamp": feature_timestamp,
            "feature_version": FEATURE_VERSION,
            "history_count": len(driver_history),
            "form_window": form_window,
            "session_source": session_source if practice else None,
            "provenance": provenance,
            "qualifying_position": q_position,
            "qualifying_last_session_seconds": next(
                (value for value in q_times if value is not None), None
            ),
            "grid_position": _number(entrant.get("grid_position"), "published grid", positive=True),
            "teammate_qualifying_position_delta": q_position - teammate_position
            if q_position is not None and teammate_position is not None
            else None,
            "recent_finish_mean": _average(
                [
                    _number(value.get("position"), "finish position", positive=True)
                    for value in driver_history
                ]
            ),
            "recent_dnf_rate": mean(dnf_values) if dnf_values else None,
            "constructor_recent_finish_mean": _average(
                [
                    _number(value.get("position"), "constructor finish", positive=True)
                    for value in constructor_history
                ]
            ),
            "recent_pit_stop_seconds": _average(
                [
                    _number(value.get("pit_stop_seconds"), "pit stop", positive=True)
                    for value in driver_history
                ]
            ),
            "practice_best_seconds": min(best_present) if best_present else None,
            "practice_median_seconds": _average(
                [
                    _number(value.get("median_lap_seconds"), "practice median", positive=True)
                    for value in practice
                ]
            ),
            "tyre_age_mean": _average(
                [_number(value.get("median_tyre_age"), "tyre age") for value in practice]
            ),
            "tyre_compound_count": float(len(compounds)) if compounds else None,
            "driver_championship_points": _number(standings.get("driver_points"), "driver points"),
            "constructor_championship_points": _number(
                standings.get("constructor_points"), "constructor points"
            ),
            "circuit_length_km": _number(
                circuit_row.get("length_km"), "circuit length", positive=True
            )
            if circuit_row
            else None,
            "is_street_circuit": float(circuit_row["is_street"])
            if circuit_row and isinstance(circuit_row.get("is_street"), bool)
            else None,
        }
        for variable in ("temperature_2m", "precipitation_probability", "wind_speed_10m"):
            value = weather_row.get(variable) if weather_row else None
            # Temperature is the only signed physical input in this feature set.
            if variable == "temperature_2m" and value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(value)
                ):
                    raise ValueError("forecast temperature must be finite")
                row[f"forecast_{variable}"] = float(value)
            else:
                row[f"forecast_{variable}"] = _number(value, variable)
        probability = row["forecast_precipitation_probability"]
        if probability is not None and probability > 100:
            raise ValueError("precipitation_probability must be between 0 and 100")
        for name in NUMERIC_FEATURES:
            row[f"{name}_missing"] = row[name] is None
        output.append(row)
    return pa.Table.from_pylist(output, schema=FEATURE_SCHEMA)
