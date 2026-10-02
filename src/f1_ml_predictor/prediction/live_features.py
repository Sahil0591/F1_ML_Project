"""Point-in-time feature rows for an upcoming race, frozen before any prediction.

Rows reuse the exact Gold rolling, constructor-form and scoring-ledger rules on
audited history published before the cutoff. Inputs that do not exist at the
cutoff stay missing; nothing is filled from later results, grids or weather.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import BENCHMARK_FEATURE_COLUMNS, file_sha256
from f1_ml_predictor.benchmarks.enrichment import _PRACTICE_FEATURES, _derive
from f1_ml_predictor.benchmarks.rolling import WINDOWS, rolling_driver_features
from f1_ml_predictor.benchmarks.scoring import _POINT_FEATURES as SCORING_POINT_FEATURES
from f1_ml_predictor.benchmarks.scoring import _proof
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.normalization.jolpica import normalize_schedule
from f1_ml_predictor.prediction.history import GoldVersion
from f1_ml_predictor.scoring.ledger import ScoringLedger
from f1_ml_predictor.time import require_known_by, require_utc

LIVE_SNAPSHOT_VERSION = "development-live-snapshot-v1"
PRE_QUALIFYING = "pre_qualifying"
SAME_WEEKEND_FEATURES = (
    "qualifying_position",
    "qualifying_last_session_seconds",
    "grid_position",
    "teammate_qualifying_position_delta",
)
_CONTEXT = (
    "event_id",
    "driver_id",
    "constructor_id",
    "circuit_id",
    "prediction_timestamp",
    "feature_timestamp",
    "benchmark_tier",
    "cutoff_kind",
    "qualifying_status",
    "start_type",
    "pit_lane_start",
    "grid_status",
)
_SNAPSHOT_DIRECTORY = Path("data/features/development_snapshots")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class ScheduledEvent:
    event: EventId
    race_name: str
    circuit_id: str
    race_start: datetime
    qualifying_start: datetime | None
    sprint_start: datetime | None


def _session_start(race: dict[str, Any], name: str) -> datetime | None:
    session = race.get(name)
    if not isinstance(session, dict) or not session.get("date") or not session.get("time"):
        return None
    start = datetime.fromisoformat(f"{session['date']}T{session['time']}")
    require_utc(start, f"{name} start")
    return start


def load_schedule(root: Path, observation: str) -> tuple[list[ScheduledEvent], dict[str, Any]]:
    """Read one retained collector schedule response after checking its payload hash."""
    path = (root / observation).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("schedule observation escapes the project root")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("role") != "schedule" or record.get("provider") != "jolpica":
        raise ValueError("observation is not a retained Jolpica schedule response")
    if _digest(_canonical(record["payload"])) != record["payload_sha256"]:
        raise ValueError("retained schedule payload hash mismatch")
    captured = datetime.fromisoformat(record["response_captured_at"])
    require_utc(captured, "schedule capture time")
    races = record["payload"]["MRData"]["RaceTable"]["Races"]
    season = int(record["payload"]["MRData"]["RaceTable"]["season"])
    normalized = {row["round"]: row for row in normalize_schedule(races, season).to_pylist()}
    events = []
    for race in races:
        row = normalized[int(race["round"])]
        if row["race_start_utc"] is None:
            continue
        events.append(
            ScheduledEvent(
                EventId(season, row["round"]),
                row["race_name"],
                row["circuit_id"],
                row["race_start_utc"],
                _session_start(race, "Qualifying"),
                _session_start(race, "Sprint"),
            )
        )
    metadata = {
        "path": observation,
        "file_sha256": file_sha256(path),
        "payload_sha256": record["payload_sha256"],
        "url": record["url"],
        "captured_at": captured.isoformat(),
        "season": season,
    }
    return sorted(events, key=lambda item: item.race_start), metadata


def scoring_gate_notes(ledger: ScoringLedger, event: EventId, cutoff: datetime) -> list[str]:
    """Explain why the strict ledger gate leaves point features missing."""
    notes = []
    rule = ledger.rule_for(event)
    if rule is None or rule.revision_status not in {"audited", "revised"}:
        notes.append(f"no audited scoring rule covers {event.partition()}")
    for number in range(1, event.round):
        prior = EventId(event.season, number)
        versions = [
            version
            for version in ledger.events
            if version.event == prior
            and version.completed_at < cutoff
            and version.effective_at < cutoff
        ]
        if not versions:
            notes.append(f"round {number} has no points version published before the cutoff")
            break
        latest = max(versions, key=lambda version: version.effective_at)
        if (
            not latest.complete
            or latest.revision_status not in {"audited", "revised", "disputed"}
            or any(entry.total_points is None for entry in latest.entries)
        ):
            notes.append(f"round {number} points are {latest.revision_status} at the cutoff")
            break
        if latest.revision_status == "disputed":
            notes.append(f"round {number} uses published points still under appeal")
    return notes


def _default_row(version: GoldVersion) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in version.feature_columns:
        field = version.schema.field(name)
        if name.endswith("_missing"):
            values[name] = True
        elif pa.types.is_integer(field.type):
            values[name] = 0
        else:
            values[name] = None
    return values


def build_live_rows(
    version: GoldVersion,
    outcomes: dict[tuple[int, int], str],
    ledger: ScoringLedger,
    *,
    target: ScheduledEvent,
    cutoff: datetime,
    cutoff_kind: str,
    roster: dict[str, str],
    opened_at: datetime,
    base: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive one complete roster of label-free rows and per-driver provenance."""
    require_utc(cutoff, "prediction cutoff")
    require_known_by(opened_at, cutoff)
    if not cutoff < target.race_start:
        raise ValueError("prediction cutoff must precede the race start")
    if base is not None and set(base) != set(roster):
        raise ValueError("certified capture roster differs from the prediction roster")
    events = {
        key: rows
        for key, rows in version.events.items()
        if all(row["prediction_timestamp"] < cutoff for row in rows)
    }
    season, round_number = target.event.season, target.event.round
    rows: list[dict[str, Any]] = []
    for driver, constructor in sorted(roster.items()):
        row = {
            **_default_row(version),
            "event_id": target.event.partition(),
            "driver_id": driver,
            "constructor_id": constructor,
            "circuit_id": target.circuit_id,
            "prediction_timestamp": cutoff,
            "feature_timestamp": opened_at,
            "benchmark_tier": "Development",
            "cutoff_kind": cutoff_kind,
            "qualifying_status": "not_held_before_cutoff",
            "start_type": "unknown",
            "pit_lane_start": None,
            "grid_status": "unknown",
        }
        if base is not None:
            captured = base[driver]
            if captured["prediction_timestamp"] != cutoff:
                raise ValueError("certified capture cutoff differs from the prediction cutoff")
            require_known_by(captured["feature_timestamp"], cutoff)
            for name in (*BENCHMARK_FEATURE_COLUMNS, *_CONTEXT[8:]):
                row[name] = captured[name]
            row["feature_timestamp"] = max(opened_at, captured["feature_timestamp"])
        rows.append(row)
    provenance = []
    enriched_rows = []
    gate = scoring_gate_notes(ledger, target.event, cutoff)
    standings = ledger.standings_before(target.event, cutoff, roster)
    history = _proof(ledger, target.event, cutoff)
    for row in rows:
        driver = row["driver_id"]
        rolling, proofs = rolling_driver_features(
            events, outcomes, season, round_number, driver, cutoff
        )
        row.update(rolling)
        clocks = [row["feature_timestamp"]]
        for window in WINDOWS:
            for name in (f"recent_finish_mean_{window}", f"recent_dnf_rate_{window}"):
                row[f"{name}_missing"] = row[name] is None
            if proofs[str(window)]["complete"]:
                clocks.extend(
                    historic["label_available_at"]
                    for key in [
                        (season, number) for number in range(round_number - window, round_number)
                    ]
                    for historic in events[key]
                )
        row["feature_timestamp"] = max(clocks)
        enriched, derived = _derive(row, rows, events, outcomes)
        values = standings[driver]
        reasons: dict[str, str] = dict(derived["reasons"])
        for name in _PRACTICE_FEATURES:
            if enriched[name] is None:
                reasons[name] = "target_weekend_practice_not_in_live_snapshot"
        for name in SCORING_POINT_FEATURES:
            enriched[name] = values[name]
            enriched[f"{name}_missing"] = values[name] is None
            if values[name] is None:
                reasons[name] = str(
                    values["missing_reason"]
                    or (
                        values.get("constructor_missing_reason")
                        if name.startswith("constructor_")
                        else None
                    )
                    or (
                        "insufficient_prior_rounds"
                        if "last_" in name
                        else "unresolved_championship_countback_tie"
                    )
                )
            else:
                reasons.pop(name, None)
        if values["missing_reason"] is None and history:
            enriched["feature_timestamp"] = max(
                enriched["feature_timestamp"],
                *(datetime.fromisoformat(record["effective_at"]) for record in history),
            )
        for window in WINDOWS:
            if rolling[f"recent_finish_mean_{window}"] is None:
                reasons[f"recent_finish_mean_{window}"] = (
                    "driver_has_no_classified_finish_in_window"
                    if proofs[str(window)]["complete"]
                    else "prior_window_incomplete_or_unaudited"
                )
            if rolling[f"recent_dnf_rate_{window}"] is None:
                reasons[f"recent_dnf_rate_{window}"] = "no_audited_binary_dnf_labels_in_window"
        for name in BENCHMARK_FEATURE_COLUMNS:
            if name.endswith("_missing") or enriched[name] is not None:
                continue
            reasons[name] = (
                "qualifying_not_held_before_cutoff"
                if base is None and name in SAME_WEEKEND_FEATURES
                else "not_in_certified_capture"
                if base is not None and name in SAME_WEEKEND_FEATURES
                else "no_point_in_time_source_in_live_snapshot"
            )
        require_known_by(enriched["feature_timestamp"], cutoff)
        enriched_rows.append(enriched)
        provenance.append(
            {
                "driver_id": driver,
                "constructor_id": row["constructor_id"],
                "feature_available_at": enriched["feature_timestamp"].isoformat(),
                "missing_reasons": dict(sorted(reasons.items())),
                "rolling_windows": proofs,
                "constructor_form_sources": derived["source_events"],
                "teammate_driver_ids": derived["teammate_driver_ids"],
                "scoring_gate": gate,
            }
        )
    return enriched_rows, provenance


def availability_mask(version: GoldVersion, live_rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Name trained predictors that no live row has at this cutoff."""
    paired = [
        name
        for name in version.feature_columns
        if not name.endswith("_missing") and f"{name}_missing" in version.feature_columns
    ]
    return tuple(
        name
        for name in paired
        if all(row[name] is None for row in live_rows)
        and any(row[name] is not None for row in version.rows)
    )


def apply_mask(rows: list[dict[str, Any]], masked: tuple[str, ...]) -> list[dict[str, Any]]:
    """Hide unavailable predictors from training so they are not median-imputed live."""
    hidden = {name: None for name in masked} | {f"{name}_missing": True for name in masked}
    return [{**row, **{key: value for key, value in hidden.items() if key in row}} for row in rows]


def missing_summary(
    version: GoldVersion,
    live_rows: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    masked: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Report every trained predictor's live coverage, training coverage and reasons."""
    summary = []
    for name in version.feature_columns:
        if name.endswith("_missing"):
            continue
        reasons = sorted(
            {
                record["missing_reasons"][name]
                for record in provenance
                if name in record["missing_reasons"]
            }
        )
        summary.append(
            {
                "feature": name,
                "live_present": sum(row[name] is not None for row in live_rows),
                "live_rows": len(live_rows),
                "training_present": sum(row[name] is not None for row in version.rows),
                "masked_for_training": name in masked,
                "reasons": reasons,
            }
        )
    return summary


def freeze_snapshot(
    root: Path,
    version: GoldVersion,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Write content-addressed Parquet and manifest; identical retries verify bytes."""
    fields = [version.schema.field(name) for name in _CONTEXT]
    fields.extend(version.schema.field(name) for name in version.feature_columns)
    schema = pa.schema([pa.field(field.name, field.type) for field in fields])
    table = pa.Table.from_pylist(
        [{field.name: row[field.name] for field in schema} for row in rows], schema=schema
    )
    event = rows[0]["event_id"]
    directory = root / _SNAPSHOT_DIRECTORY / event / rows[0]["cutoff_kind"]
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / ".pending.parquet"
    pq.write_table(table, pending)
    digest = file_sha256(pending)
    path = directory / f"{digest}.parquet"
    if path.exists():
        pending.unlink()
        if file_sha256(path) != digest:
            raise ValueError("immutable development snapshot collision")
    else:
        pending.replace(path)
    manifest = {
        "version": LIVE_SNAPSHOT_VERSION,
        "validation_status": "development_only",
        "snapshot_path": path.relative_to(root).as_posix(),
        "snapshot_sha256": digest,
        **metadata,
    }
    manifest_bytes = _canonical(manifest)
    manifest_path = directory / f"{_digest(manifest_bytes)}.json"
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("immutable development snapshot manifest collision")
    else:
        with manifest_path.open("xb") as handle:
            handle.write(manifest_bytes)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": _digest(manifest_bytes),
    }
