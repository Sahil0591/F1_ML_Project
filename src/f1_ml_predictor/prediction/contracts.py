"""Cutoff-specific feature contracts shared by historical datasets and live snapshots.

Each named cutoff has its own feature list. History features are recomputed at
the contract cutoff from audited Gold outcomes, audited binary DNF labels and the
scoring ledger, using only values published before that cutoff. Weekend features
(practice, qualifying, grid) are added only by the contracts that permit them and
only when their publication precedes the cutoff. The same function builds
historical rows and live rows, so training and prediction share one code path.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.benchmarks.enrichment import _PRACTICE_FEATURES, _derive
from f1_ml_predictor.benchmarks.rolling import rolling_driver_features
from f1_ml_predictor.benchmarks.scoring import _POINT_FEATURES as POINT_FEATURES
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.prediction.history import GoldVersion
from f1_ml_predictor.prediction.protocol import CUTOFFS
from f1_ml_predictor.prediction.schedules import Weekend
from f1_ml_predictor.prediction.strength_features import (
    CIRCUIT_PROFILES,
    STRENGTH_COUNTS,
    STRENGTH_NUMERIC,
    elo_ratings,
    similar_circuit_delta,
    teammate_head_to_head,
)
from f1_ml_predictor.scoring.ledger import ScoringLedger
from f1_ml_predictor.time import require_known_by, require_utc

CONTRACT_VERSION = "cutoff-contracts-v3"
HISTORY_NUMERIC = (
    "recent_finish_mean_3",
    "recent_finish_mean_5",
    "recent_finish_mean_10",
    "driver_finish_mean_any_5",
    "driver_finish_mean_any_10",
    "driver_qualifying_mean_any_5",
    "driver_dnf_rate_any_10",
    "constructor_dnf_rate_any_10",
    "constructor_average_finish_last_3",
    "constructor_average_finish_last_5",
    "constructor_average_finish_last_10",
    "constructor_qualifying_form",
    "constructor_teammate_aggregated_form",
    *POINT_FEATURES,
    "driver_circuit_finish_mean",
    "circuit_dnf_rate",
    *STRENGTH_NUMERIC,
)
HISTORY_COUNTS = (
    "history_count_3",
    "history_count_5",
    "history_count_10",
    "driver_finish_observations_any_10",
    "driver_dnf_observations_any_10",
    "constructor_dnf_observations_any_10",
    "driver_circuit_observations",
    "circuit_dnf_observations",
    "circuit_seen_before",
    "sprint_weekend",
    *STRENGTH_COUNTS,
)
PRACTICE_NUMERIC = tuple(_PRACTICE_FEATURES)
QUALIFYING_NUMERIC = (
    "qualifying_position",
    "qualifying_last_session_seconds",
    "teammate_qualifying_position_delta",
)
GRID_NUMERIC = ("grid_position",)
# Same-weekend sprint values (v3). Since 2024 the sprint precedes Grand Prix
# qualifying, so the post-qualifying cutoffs see it; earlier formats and non-sprint
# weekends leave them missing, which the sprint_weekend count disambiguates.
SPRINT_NUMERIC = ("sprint_qualifying_position", "sprint_position", "sprint_classified")
_WEEKEND = {
    "pre_weekend": (),
    "post_practice": PRACTICE_NUMERIC,
    "post_qualifying": (*PRACTICE_NUMERIC, *QUALIFYING_NUMERIC, *SPRINT_NUMERIC),
    "pre_race": (*PRACTICE_NUMERIC, *QUALIFYING_NUMERIC, *GRID_NUMERIC, *SPRINT_NUMERIC),
}
DNF_FEATURES = (
    "driver_dnf_rate_any_10",
    "constructor_dnf_rate_any_10",
    "circuit_dnf_rate",
    "driver_dnf_observations_any_10",
    "constructor_dnf_observations_any_10",
    "circuit_dnf_observations",
    "circuit_seen_before",
    "sprint_weekend",
)
LABELS = (
    "label_position",
    "label_winner",
    "label_podium",
    "label_dnf",
    "label_available_at",
    "label_dnf_available_at",
)
_CONTEXT = (
    "event_id",
    "driver_id",
    "constructor_id",
    "circuit_id",
    "prediction_timestamp",
    "feature_timestamp",
    "cutoff_kind",
)


def numeric_features(contract: str) -> tuple[str, ...]:
    if contract not in CUTOFFS:
        raise ValueError(f"unknown cutoff contract {contract}")
    return (*HISTORY_NUMERIC, *_WEEKEND[contract])


def feature_columns(contract: str) -> tuple[str, ...]:
    """Ordered predictors: numeric values, their missing flags, then counts."""
    numeric = numeric_features(contract)
    return (*numeric, *(f"{name}_missing" for name in numeric), *HISTORY_COUNTS)


def dnf_columns(contract: str) -> tuple[str, ...]:
    """Reliability predictors; qualifying is added once the contract permits it."""
    numeric = [name for name in DNF_FEATURES if name in HISTORY_NUMERIC]
    if "qualifying_position" in numeric_features(contract):
        numeric.append("qualifying_position")
    counts = [name for name in DNF_FEATURES if name in HISTORY_COUNTS]
    return (*numeric, *(f"{name}_missing" for name in numeric), *counts)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _event_key(event_id: str) -> tuple[int, int]:
    season, number = event_id.split("/")
    return int(season.removeprefix("season=")), int(number.removeprefix("round="))


@dataclass
class AuditedHistory:
    """Gold rows with merged binary DNF labels, the ledger and retained schedules."""

    version: GoldVersion
    outcomes: dict[tuple[int, int], str]
    ledger: ScoringLedger
    weekends: dict[EventId, Weekend]
    dnf_labels: dict[tuple[str, str], tuple[bool, datetime]]
    # Gold sprint rows by event and driver, and the sprint version's manifest hash.
    sprints: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    sprint_version: str | None = None
    rows: list[dict[str, Any]] = field(init=False)
    by_event: dict[str, list[dict[str, Any]]] = field(init=False)
    order: list[str] = field(init=False)

    def __post_init__(self) -> None:
        rows = []
        for row in self.version.rows:
            label = self.dnf_labels.get((row["event_id"], row["driver_id"]))
            rows.append(
                {
                    **row,
                    "label_dnf": None if label is None else label[0],
                    "label_dnf_available_at": None if label is None else label[1],
                }
            )
        self.rows = rows
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["event_id"]].append(row)
        self.by_event = dict(grouped)
        self.order = sorted(
            grouped, key=lambda event: (grouped[event][0]["prediction_timestamp"], event)
        )


def _race_time(history: AuditedHistory, event: EventId) -> datetime:
    weekend = history.weekends.get(event)
    if weekend is None or weekend.race is None:
        raise ValueError(f"no retained race start for {event.partition()}")
    return weekend.race


def history_features(
    history: AuditedHistory,
    event: EventId,
    circuit_id: str,
    cutoff: datetime,
    roster: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], datetime]:
    """Compute every history predictor for one roster at one cutoff.

    Only earlier Gold races whose labels were published before the cutoff are
    read. DNF labels additionally require their own publication before the cutoff.
    """
    require_utc(cutoff, "contract cutoff")
    target_start = _race_time(history, event)
    prior = [
        name
        for name in history.order
        if name != event.partition()
        and _race_time(history, EventId(*_event_key(name))) < target_start
        and all(row["label_available_at"] < cutoff for row in history.by_event[name])
    ]
    events = {_event_key(name): history.by_event[name] for name in prior}
    latest_first = list(reversed(prior))
    clocks = [
        max(
            (row["label_available_at"] for name in prior for row in history.by_event[name]),
            default=cutoff,
        )
    ]
    season, round_number = event.season, event.round
    current: list[dict[str, Any]] = [
        {
            "event_id": event.partition(),
            "driver_id": driver,
            "constructor_id": constructor,
            "prediction_timestamp": cutoff,
            "feature_timestamp": cutoff,
        }
        for driver, constructor in sorted(roster.items())
    ]
    standings = history.ledger.standings_before(event, cutoff, roster)
    circuit_events = [
        name for name in latest_first if history.by_event[name][0]["circuit_id"] == circuit_id
    ]
    circuit_dnf = [
        row["label_dnf"]
        for name in circuit_events
        for row in history.by_event[name]
        if row["label_dnf"] is not None and row["label_dnf_available_at"] < cutoff
    ]
    weekend = history.weekends.get(event)
    sprint = int(weekend is not None and weekend.sprint is not None)
    driver_elo, constructor_elo, elo_events = elo_ratings(
        [history.by_event[name] for name in prior], cutoff
    )
    values: dict[str, dict[str, Any]] = {}
    reasons: dict[str, dict[str, Any]] = {}
    for row in current:
        driver, constructor = row["driver_id"], row["constructor_id"]
        rolling, proofs = rolling_driver_features(
            events, history.outcomes, season, round_number, driver, cutoff
        )
        derived, derived_proof = _derive(row, current, events, history.outcomes)
        appearances = [
            item
            for name in latest_first
            for item in history.by_event[name]
            if item["driver_id"] == driver
        ]
        finishes = [
            float(item["label_position"])
            for item in appearances[:10]
            if item["label_position"] is not None
        ]
        qualifying = [
            float(item["qualifying_position"])
            for item in appearances[:5]
            if item["qualifying_position"] is not None
        ]
        driver_dnf = [
            item["label_dnf"]
            for item in appearances
            if item["label_dnf"] is not None and item["label_dnf_available_at"] < cutoff
        ][:10]
        team_events = [
            name
            for name in latest_first
            if any(item["constructor_id"] == constructor for item in history.by_event[name])
        ][:10]
        team_dnf = [
            item["label_dnf"]
            for name in team_events
            for item in history.by_event[name]
            if item["constructor_id"] == constructor
            and item["label_dnf"] is not None
            and item["label_dnf_available_at"] < cutoff
        ]
        at_circuit = [
            float(item["label_position"])
            for name in circuit_events
            for item in history.by_event[name]
            if item["driver_id"] == driver and item["label_position"] is not None
        ]
        team_history = [
            (item["circuit_id"], float(item["label_position"]))
            for name in latest_first
            for item in history.by_event[name]
            if item["constructor_id"] == constructor and item["label_position"] is not None
        ]
        points = standings[driver]
        record: dict[str, Any] = {
            "recent_finish_mean_3": rolling["recent_finish_mean_3"],
            "recent_finish_mean_5": rolling["recent_finish_mean_5"],
            "recent_finish_mean_10": rolling["recent_finish_mean_10"],
            "history_count_3": rolling["history_count_3"],
            "history_count_5": rolling["history_count_5"],
            "history_count_10": rolling["history_count_10"],
            "driver_finish_mean_any_5": mean(finishes[:5]) if finishes[:5] else None,
            "driver_finish_mean_any_10": mean(finishes) if finishes else None,
            "driver_finish_observations_any_10": len(finishes),
            "driver_qualifying_mean_any_5": mean(qualifying) if qualifying else None,
            "driver_dnf_rate_any_10": mean(map(float, driver_dnf)) if driver_dnf else None,
            "driver_dnf_observations_any_10": len(driver_dnf),
            "constructor_dnf_rate_any_10": mean(map(float, team_dnf)) if team_dnf else None,
            "constructor_dnf_observations_any_10": len(team_dnf),
            "driver_circuit_finish_mean": mean(at_circuit) if at_circuit else None,
            "driver_circuit_observations": len(at_circuit),
            "circuit_dnf_rate": mean(map(float, circuit_dnf)) if circuit_dnf else None,
            "circuit_dnf_observations": len(circuit_dnf),
            "circuit_seen_before": int(bool(circuit_events)),
            "sprint_weekend": sprint,
            "driver_elo": (driver_elo[driver] - 1500.0 if driver in driver_elo else None),
            "constructor_elo": (
                constructor_elo[constructor] - 1500.0 if constructor in constructor_elo else None
            ),
            "driver_elo_events": elo_events.get(driver, 0),
            "driver_teammate_qualifying_h2h_10": teammate_head_to_head(
                appearances, history.by_event
            ),
            "driver_similar_circuit_delta": similar_circuit_delta(
                [
                    (item["circuit_id"], float(item["label_position"]))
                    for item in appearances
                    if item["label_position"] is not None
                ],
                circuit_id,
            ),
            "constructor_similar_circuit_delta": similar_circuit_delta(
                team_history, circuit_id, limit=80
            ),
        }
        for name in (
            "constructor_average_finish_last_3",
            "constructor_average_finish_last_5",
            "constructor_average_finish_last_10",
            "constructor_qualifying_form",
            "constructor_teammate_aggregated_form",
        ):
            record[name] = derived[name]
        for name in POINT_FEATURES:
            record[name] = points[name]
        missing = {
            name: (
                points.get("missing_reason") or "unresolved_championship_countback_tie"
                if name in POINT_FEATURES
                else derived_proof["reasons"].get(name)
                or ("prior_window_incomplete_or_unaudited" if name.startswith("recent_") else None)
                or "no_audited_observation_before_cutoff"
            )
            for name in HISTORY_NUMERIC
            if record[name] is None
        }
        if circuit_id not in CIRCUIT_PROFILES:
            for name in ("driver_similar_circuit_delta", "constructor_similar_circuit_delta"):
                missing[name] = "circuit_unprofiled"
        if not circuit_events:
            missing["driver_circuit_finish_mean"] = "circuit_unseen_in_gold_history"
            missing["circuit_dnf_rate"] = "circuit_unseen_in_gold_history"
        values[driver] = record
        reasons[driver] = {
            "missing_reasons": dict(sorted(missing.items())),
            "points_status": points.get("points_status"),
            "rolling_windows": proofs,
        }
    if standings and next(iter(standings.values())).get("missing_reason") is None:
        history_proof = [
            version.effective_at
            for version in history.ledger.events
            if version.event.season == season
            and version.event.round < round_number
            and version.effective_at < cutoff
        ]
        clocks.extend(history_proof)
    clocks.extend(
        row["label_dnf_available_at"]
        for name in prior
        for row in history.by_event[name]
        if row["label_dnf_available_at"] is not None and row["label_dnf_available_at"] < cutoff
    )
    available = min(max(clocks), cutoff)
    require_known_by(available, cutoff)
    return values, reasons, available


def contract_cutoff(history: AuditedHistory, contract: str, event_id: str) -> datetime:
    """Historical cutoff placement for one Gold race and contract."""
    rows = history.by_event[event_id]
    weekend = history.weekends[EventId(*_event_key(event_id))]
    if weekend.circuit_id != rows[0]["circuit_id"]:
        raise ValueError(f"{event_id} schedule circuit differs from the audited Gold circuit")
    post_qualifying: datetime = rows[0]["prediction_timestamp"]
    if contract == "pre_weekend":
        cutoff = weekend.first_practice
    elif contract == "post_practice":
        cutoff = weekend.qualifying
    else:
        cutoff = post_qualifying
    if cutoff is None or cutoff > post_qualifying:
        raise ValueError(f"{event_id} has no valid {contract} cutoff")
    return cutoff


def build_rows(
    history: AuditedHistory,
    contract: str,
    *,
    event: EventId,
    circuit_id: str,
    cutoff: datetime,
    roster: dict[str, str],
    weekend: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One label-free row per driver for a contract; weekend values need their clocks."""
    values, reasons, available = history_features(history, event, circuit_id, cutoff, roster)
    allowed = _WEEKEND[contract]
    rows = []
    for driver, constructor in sorted(roster.items()):
        row: dict[str, Any] = {
            "event_id": event.partition(),
            "driver_id": driver,
            "constructor_id": constructor,
            "circuit_id": circuit_id,
            "prediction_timestamp": cutoff,
            "feature_timestamp": available,
            "cutoff_kind": contract,
            **values[driver],
        }
        source = (weekend or {}).get(driver, {})
        for name in allowed:
            clock = source.get(f"{name}_available_at")
            value = source.get(name)
            if value is not None and clock is not None and clock <= cutoff:
                row[name] = float(value)
                row["feature_timestamp"] = max(row["feature_timestamp"], clock)
            else:
                row[name] = None
                reasons[driver]["missing_reasons"][name] = (
                    "weekend_source_absent" if value is None else "published_after_cutoff"
                )
        for name in numeric_features(contract):
            row[f"{name}_missing"] = row[name] is None
        require_known_by(row["feature_timestamp"], cutoff)
        rows.append(row)
    return rows, reasons


def historical_weekend(
    history: AuditedHistory, event_id: str, practice_clock: dict[tuple[str, str], datetime]
) -> dict[str, dict[str, Any]]:
    """Weekend values from a Gold race with their audited publication bounds."""
    result = {}
    for row in history.by_event[event_id]:
        cutoff = row["prediction_timestamp"]
        practice = practice_clock.get((event_id, row["driver_id"]))
        item: dict[str, Any] = {}
        for name in PRACTICE_NUMERIC:
            item[name] = row[name]
            item[f"{name}_available_at"] = practice
        for name in (*QUALIFYING_NUMERIC, *GRID_NUMERIC):
            item[name] = row[name]
            item[f"{name}_available_at"] = row["feature_timestamp"]
            if row["feature_timestamp"] > cutoff:
                raise ValueError("Gold weekend value is later than its audited cutoff")
        item.update(sprint_weekend_values(history.sprints.get(event_id, {}).get(row["driver_id"])))
        result[row["driver_id"]] = item
    return result


def sprint_weekend_values(row: dict[str, Any] | None) -> dict[str, Any]:
    """Race-contract sprint values from one Gold sprint row, each with its own clock."""
    if row is None:
        return {}
    return {
        "sprint_qualifying_position": row["sprint_qualifying_position"],
        "sprint_qualifying_position_available_at": row["grid_available_at"],
        "sprint_position": row["label_position"],
        "sprint_position_available_at": row["label_available_at"],
        "sprint_classified": float(row["label_classified"]),
        "sprint_classified_available_at": row["label_available_at"],
    }


def practice_clocks(version: GoldVersion) -> dict[tuple[str, str], datetime]:
    return practice_clocks_from(version.directory)


def practice_clocks_from(directory: Path) -> dict[tuple[str, str], datetime]:
    """Audited practice publication bounds by event and driver for one Gold version."""
    records = json.loads((directory / "feature_provenance.json").read_text("utf-8"))
    return {
        (item["event_id"], item["driver_id"]): datetime.fromisoformat(
            item["practice"]["available_at"]
        )
        for item in records
        if "practice" in item
    }


def dnf_label_map(dnf_version: GoldVersion) -> dict[tuple[str, str], tuple[bool, datetime]]:
    return {
        (row["event_id"], row["driver_id"]): (bool(row["label_dnf"]), row["label_available_at"])
        for row in dnf_version.rows
        if row["label_dnf"] is not None
    }


def _schema(contract: str) -> pa.Schema:
    timestamp = pa.timestamp("us", tz="UTC")
    fields = [
        pa.field("event_id", pa.string()),
        pa.field("driver_id", pa.string()),
        pa.field("constructor_id", pa.string()),
        pa.field("circuit_id", pa.string()),
        pa.field("prediction_timestamp", timestamp),
        pa.field("feature_timestamp", timestamp),
        pa.field("cutoff_kind", pa.string()),
    ]
    for name in feature_columns(contract):
        kind = (
            pa.bool_()
            if name.endswith("_missing")
            else pa.int32()
            if name in HISTORY_COUNTS
            else pa.float64()
        )
        fields.append(pa.field(name, kind))
    fields += [
        pa.field("label_position", pa.int64()),
        pa.field("label_winner", pa.bool_()),
        pa.field("label_podium", pa.bool_()),
        pa.field("label_dnf", pa.bool_()),
        pa.field("label_available_at", timestamp),
        pa.field("label_dnf_available_at", timestamp),
    ]
    return pa.schema(fields)


def build_contract_datasets(
    root: Path,
    history: AuditedHistory,
    dnf_version: GoldVersion,
    schedule_sources: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Freeze one immutable dataset per cutoff contract for every Gold race."""
    clocks = practice_clocks(history.version)
    output = (
        root
        / "data/benchmarks"
        / f"gold_{CONTRACT_VERSION.replace('-', '_')}"
        / history.version.manifest_sha256
        / dnf_version.manifest_sha256
        / (history.sprint_version or "no_gold_sprints")
    )
    result = {}
    for contract in CUTOFFS:
        rows = []
        for event_id in history.order:
            gold = history.by_event[event_id]
            event = EventId(*_event_key(event_id))
            cutoff = contract_cutoff(history, contract, event_id)
            roster = {row["driver_id"]: row["constructor_id"] for row in gold}
            built, _ = build_rows(
                history,
                contract,
                event=event,
                circuit_id=gold[0]["circuit_id"],
                cutoff=cutoff,
                roster=roster,
                weekend=historical_weekend(history, event_id, clocks),
            )
            labels = {row["driver_id"]: row for row in gold}
            for row in built:
                label = labels[row["driver_id"]]
                row.update({name: label[name] for name in LABELS})
                rows.append(row)
        table = pa.Table.from_pylist(rows, schema=_schema(contract))
        directory = output / contract
        directory.mkdir(parents=True, exist_ok=True)
        pending = directory / ".pending.parquet"
        pq.write_table(table, pending)
        digest = file_sha256(pending)
        path = directory / "dataset.parquet"
        if path.exists():
            pending.unlink()
            if file_sha256(path) != digest:
                raise ValueError(f"immutable {contract} contract dataset collision")
        else:
            pending.replace(path)
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "contract": contract,
            "feature_columns": list(feature_columns(contract)),
            "dnf_columns": list(dnf_columns(contract)),
            "dataset_sha256": digest,
            "rows": table.num_rows,
            "events": len(history.order),
            "source_gold_manifest_sha256": history.version.manifest_sha256,
            "source_dnf_manifest_sha256": dnf_version.manifest_sha256,
            "source_gold_sprint_manifest_sha256": history.sprint_version,
            "scoring_ledger_sha256": history.ledger.sha256,
            "schedule_sources": schedule_sources,
            "practice_policy": "practice only when its FIA publication precedes the cutoff",
        }
        manifest_path = directory / "manifest.json"
        manifest_bytes = _canonical(manifest)
        if manifest_path.exists():
            if manifest_path.read_bytes() != manifest_bytes:
                raise ValueError(f"immutable {contract} contract manifest collision")
        else:
            with manifest_path.open("xb") as handle:
                handle.write(manifest_bytes)
        result[contract] = {
            "path": path,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "rows": rows,
            "manifest": manifest,
        }
    return result
