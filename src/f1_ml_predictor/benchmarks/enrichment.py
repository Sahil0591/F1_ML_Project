"""Evidence-bounded constructor history on an immutable Gold benchmark.

This module deliberately does not infer championship points from finish ordinals.
The retained race labels do not encode sprint results, fastest-lap awards, reduced
points, or every scoring amendment. Those fields require a separate audited ledger.
"""

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import _safe_file, file_sha256
from f1_ml_predictor.benchmarks.rolling import ROLLING_VERSION, WINDOWS, _event
from f1_ml_predictor.time import require_known_by, require_utc
from f1_ml_predictor.trust.locking import advisory_lock
from f1_ml_predictor.trust.winter import _publication, registry_rows

ENRICHMENT_VERSION = "gold-historical-enrichment-v2"
_POINT_FEATURES = (
    "driver_points_before_race",
    "driver_championship_position",
    "driver_points_gap_to_leader",
    "constructor_points_before_race",
    "constructor_championship_position",
    "constructor_points_gap_to_leader",
    *(f"constructor_points_last_{window}" for window in WINDOWS),
)
_FORM_FEATURES = (
    *(f"constructor_average_finish_last_{window}" for window in WINDOWS),
    "constructor_dnf_rate",
    "constructor_qualifying_form",
    "constructor_teammate_aggregated_form",
)
_PRACTICE_FEATURES = (
    "practice_position",
    "best_lap_gap_to_fastest",
    "teammate_practice_delta",
    "session_relative_rank",
)
ENRICHMENT_NUMERIC_FEATURES = (*_POINT_FEATURES, *_FORM_FEATURES, *_PRACTICE_FEATURES)
_COUNTS = (
    *(f"constructor_finish_observations_last_{window}" for window in WINDOWS),
    "constructor_dnf_observations_last_5",
    "constructor_qualifying_observations_last_5",
    "teammate_finish_observations_last_5",
)
ENRICHMENT_FEATURE_COLUMNS = (
    *ENRICHMENT_NUMERIC_FEATURES,
    *(f"{name}_missing" for name in ENRICHMENT_NUMERIC_FEATURES),
    *_COUNTS,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _source(
    root: Path, benchmark: Path, catalog: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[int, int], str]]:
    manifest_path = benchmark / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("rolling_version") != ROLLING_VERSION or manifest.get("version") != 2:
        raise ValueError("historical enrichment requires a versioned Gold rolling benchmark")
    gold_spec = manifest["datasets"]["Gold"]
    gold_path = _safe_file(benchmark, gold_spec["path"], gold_spec["sha256"])
    if file_sha256(benchmark / "coverage.json") != manifest["coverage_sha256"]:
        raise ValueError("source benchmark coverage hash mismatch")
    if file_sha256(catalog) != manifest["catalog_sha256"]:
        raise ValueError("audited Gold registry changed since the source benchmark")
    rows = pq.read_table(gold_path).to_pylist()
    if len(rows) != gold_spec["rows"] or set(row["benchmark_tier"] for row in rows) != {"Gold"}:
        raise ValueError("source Gold rows or tier differ from the manifest")
    events = {row["event_id"] for row in rows}
    if events != set(gold_spec["events"]):
        raise ValueError("source Gold event set differs from the manifest")
    registry = json.loads(catalog.read_text(encoding="utf-8"))
    outcomes: dict[tuple[int, int], str] = {}
    for item in registry["races"]:
        key = _event(item["event_id"])
        if key in outcomes or item["event_id"] not in events:
            raise ValueError("audited registry and Gold event identities disagree")
        specification = item["outcomes"]
        _safe_file(root, specification["path"], specification["sha256"])
        outcomes[key] = specification["sha256"]
    if set(outcomes) != {_event(event) for event in events}:
        raise ValueError("some Gold events lack hash-verified audited outcomes")
    return manifest, rows, outcomes


def _prior_window(
    events: dict[tuple[int, int], list[dict[str, Any]]],
    season: int,
    round_number: int,
    window: int,
    cutoff: Any,
) -> tuple[list[tuple[tuple[int, int], list[dict[str, Any]]]], str | None]:
    if round_number <= window:
        return [], "insufficient_prior_rounds"
    prior = [(season, number) for number in range(round_number - window, round_number)]
    if any(key not in events for key in prior):
        return [], "unaudited_prior_race_in_window"
    selected = [(key, events[key]) for key in prior]
    if any(
        row["label_final_audited"] is not True
        or not row["label_audit_reference"]
        or row["label_available_at"] is None
        or row["label_available_at"] >= cutoff
        for _, event_rows in selected
        for row in event_rows
    ):
        return [], "prior_final_result_not_available_at_cutoff"
    for _, event_rows in selected:
        for row in event_rows:
            require_known_by(row["label_available_at"], cutoff)
            require_known_by(row["feature_timestamp"], cutoff)
    return selected, None


def practice_features(
    practice: dict[str, dict[str, Any]], field_size: int, driver: str, teammates: set[str]
) -> tuple[dict[str, float | None], dict[str, str]]:
    """Practice predictors for one driver from one parsed FIA practice classification."""
    values: dict[str, float | None] = {name: None for name in _PRACTICE_FEATURES}
    selected = practice.get(driver)
    if selected is None:
        return values, {
            name: "driver_absent_from_selected_practice_classification"
            for name in _PRACTICE_FEATURES
        }
    reasons: dict[str, str] = {}
    values["practice_position"] = float(selected["position"])
    values["session_relative_rank"] = (selected["position"] - 1) / (field_size - 1)
    timed = [
        entry["best_lap_seconds"]
        for entry in practice.values()
        if entry["best_lap_seconds"] is not None
    ]
    own = selected["best_lap_seconds"]
    if own is not None and timed:
        values["best_lap_gap_to_fastest"] = own - min(timed)
    else:
        reasons["best_lap_gap_to_fastest"] = "driver_has_no_printed_practice_lap"
    teammate_laps = [
        entry["best_lap_seconds"]
        for other, entry in practice.items()
        if other in teammates and entry["best_lap_seconds"] is not None
    ]
    if own is not None and teammate_laps:
        values["teammate_practice_delta"] = own - mean(teammate_laps)
    else:
        reasons["teammate_practice_delta"] = "no_current_teammate_with_printed_practice_lap"
    return values, reasons


def _derive(
    row: dict[str, Any],
    current_rows: list[dict[str, Any]],
    events: dict[tuple[int, int], list[dict[str, Any]]],
    outcomes: dict[tuple[int, int], str],
    grid_record: dict[str, Any] | None = None,
    practice_record: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    season, round_number = _event(row["event_id"])
    cutoff = row["prediction_timestamp"]
    require_utc(cutoff, "historical enrichment cutoff")
    constructor = row["constructor_id"]
    teammates = {
        member["driver_id"]
        for member in current_rows
        if member["constructor_id"] == constructor and member["driver_id"] != row["driver_id"]
    }
    values: dict[str, Any] = {name: None for name in ENRICHMENT_NUMERIC_FEATURES}
    values.update({name: 0 for name in _COUNTS})
    reasons = {
        **{name: "audited_scoring_and_sprint_ledger_absent" for name in _POINT_FEATURES},
        **{name: "fia_practice_classification_not_audited" for name in _PRACTICE_FEATURES},
    }
    proof: dict[str, Any] = {"source_events": {}, "reasons": reasons}
    used_clocks = [row["feature_timestamp"]]
    for window in WINDOWS:
        selected, unavailable = _prior_window(events, season, round_number, window, cutoff)
        name = f"constructor_average_finish_last_{window}"
        if unavailable is not None:
            reasons[name] = unavailable
            if window == 5:
                for feature in (
                    "constructor_dnf_rate",
                    "constructor_qualifying_form",
                    "constructor_teammate_aggregated_form",
                ):
                    reasons[feature] = unavailable
            continue
        proof["source_events"][str(window)] = [
            {"event_id": f"season={key[0]}/round={key[1]:02d}", "outcome_sha256": outcomes[key]}
            for key, _ in selected
        ]
        history = [historic for _, event_rows in selected for historic in event_rows]
        same_team = [historic for historic in history if historic["constructor_id"] == constructor]
        finishes = [
            float(historic["label_position"])
            for historic in same_team
            if historic["label_position"] is not None
        ]
        values[f"constructor_finish_observations_last_{window}"] = len(finishes)
        if finishes:
            values[name] = mean(finishes)
            used_clocks.extend(historic["label_available_at"] for historic in same_team)
        else:
            reasons[name] = "constructor_has_no_classified_prior_finish"
        if window != 5:
            continue
        known_dnf = [historic for historic in same_team if isinstance(historic["label_dnf"], bool)]
        values["constructor_dnf_observations_last_5"] = len(known_dnf)
        if known_dnf:
            values["constructor_dnf_rate"] = mean(
                float(historic["label_dnf"]) for historic in known_dnf
            )
            used_clocks.extend(historic["label_available_at"] for historic in known_dnf)
        else:
            reasons["constructor_dnf_rate"] = "no_audited_binary_dnf_observations"
        known_qualifying = [
            historic for historic in same_team if historic["qualifying_position"] is not None
        ]
        values["constructor_qualifying_observations_last_5"] = len(known_qualifying)
        if known_qualifying:
            values["constructor_qualifying_form"] = mean(
                float(historic["qualifying_position"]) for historic in known_qualifying
            )
            used_clocks.extend(historic["feature_timestamp"] for historic in known_qualifying)
        else:
            reasons["constructor_qualifying_form"] = "no_audited_prior_qualifying_positions"
        teammate_finishes = [
            historic
            for historic in history
            if historic["driver_id"] in teammates and historic["label_position"] is not None
        ]
        values["teammate_finish_observations_last_5"] = len(teammate_finishes)
        if teammate_finishes:
            values["constructor_teammate_aggregated_form"] = mean(
                float(historic["label_position"]) for historic in teammate_finishes
            )
            used_clocks.extend(historic["label_available_at"] for historic in teammate_finishes)
        else:
            reasons["constructor_teammate_aggregated_form"] = (
                "no_current_teammate"
                if not teammates
                else "current_teammates_have_no_classified_prior_finish"
            )
    if practice_record is not None:
        practice_values, practice_reasons = practice_features(
            practice_record["practice"], practice_record["field_size"], row["driver_id"], teammates
        )
        values.update(practice_values)
        for name in _PRACTICE_FEATURES:
            reasons.pop(name, None)
        reasons.update(practice_reasons)
        used_clocks.append(practice_record["available_at"])
    for name in ENRICHMENT_NUMERIC_FEATURES:
        values[f"{name}_missing"] = values[name] is None
    enriched = {**row, **values, "feature_timestamp": max(used_clocks)}
    require_known_by(enriched["feature_timestamp"], cutoff)
    proof["feature_available_at"] = enriched["feature_timestamp"].isoformat()
    proof["constructor_id"] = constructor
    proof["teammate_driver_ids"] = sorted(teammates)
    if practice_record is not None:
        proof["practice"] = {
            "evidence_class": "source_published_timestamp",
            "document_id": practice_record["document_id"],
            "document": practice_record["document"],
            "registry": practice_record["registry"],
            "available_at": practice_record["available_at"].isoformat(),
            "session": practice_record["session"],
            "field_size": practice_record["field_size"],
        }
    if grid_record is not None:
        grid_value = grid_record["grid"][row["driver_id"]]
        enriched["grid_position"] = (
            float(grid_value["grid_position"]) if grid_value["grid_position"] is not None else None
        )
        enriched["grid_position_missing"] = enriched["grid_position"] is None
        enriched["grid_status"] = grid_record["grid_status"]
        enriched["pit_lane_start"] = grid_value["pit_lane_start"]
        enriched["start_type"] = "pit_lane" if grid_value["pit_lane_start"] else "grid"
        enriched["feature_timestamp"] = max(
            enriched["feature_timestamp"], grid_record["available_at"]
        )
        require_known_by(enriched["feature_timestamp"], cutoff)
        proof["feature_available_at"] = enriched["feature_timestamp"].isoformat()
        proof["grid"] = {
            "evidence_class": "source_published_timestamp",
            "document_id": grid_record["document_id"],
            "document": grid_record["document"],
            "registry": grid_record["registry"],
            "available_at": grid_record["available_at"].isoformat(),
            "status": grid_record["grid_status"],
        }
    return enriched, proof


def _grids(
    root: Path,
    path: Path,
    source_gold_sha256: str,
    events: dict[tuple[int, int], list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], str]:
    from f1_ml_predictor.trust.grid_history import parse_grid_pdf

    digest = file_sha256(path)
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("source_gold_sha256") != source_gold_sha256 or audit.get("version") != 1:
        raise ValueError("grid audit targets another Gold source")
    records = audit["races"]
    if len(records) != len(events) or len({record["event_id"] for record in records}) != len(
        records
    ):
        raise ValueError("grid audit does not cover every Gold race exactly once")
    accepted: dict[str, dict[str, Any]] = {}
    reasons = {}
    for record in records:
        key = _event(record["event_id"])
        if key not in events:
            raise ValueError("grid audit includes a non-Gold event")
        cutoff = events[key][0]["prediction_timestamp"]
        if record["cutoff"] != cutoff.isoformat():
            raise ValueError("grid audit cutoff differs from the source benchmark")
        if record["status"] != "audited":
            reasons[record["event_id"]] = record["reason"]
            continue
        document = _safe_file(root, record["document"]["path"], record["document"]["sha256"])
        registry = _safe_file(root, record["registry"]["path"], record["registry"]["sha256"])
        matching = [
            row
            for row in registry_rows(registry.read_text(encoding="utf-8"))
            if row["url"] == record["url"]
            and row["document_id"] == record["document_id"]
            and row["publication_cet"] == record["publication_cet"]
            and not row["recalled"]
            and "starting grid" in row["title"].lower()
        ]
        if len(matching) != 1:
            raise ValueError("grid audit lacks an exact retained FIA registry row")
        available = _publication(matching[0]) + timedelta(minutes=1)
        if available.isoformat() != record["available_at"]:
            raise ValueError("grid publication bound differs from the FIA registry")
        require_known_by(available, cutoff)
        expected = {row["driver_id"] for row in events[key]}
        parsed = parse_grid_pdf(document, expected)
        if parsed != record["grid"] or set(parsed) != expected:
            raise ValueError("grid audit values differ from the exact retained PDF")
        accepted[record["event_id"]] = {**record, "available_at": available}
    return accepted, reasons, digest


def _practices(
    root: Path,
    path: Path,
    source_gold_sha256: str,
    events: dict[tuple[int, int], list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], str]:
    from f1_ml_predictor.trust.practice_history import parse_practice_pdf

    digest = file_sha256(path)
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("source_gold_sha256") != source_gold_sha256 or audit.get("version") != 1:
        raise ValueError("practice audit targets another Gold source")
    records = audit["races"]
    if len(records) != len(events) or len({record["event_id"] for record in records}) != len(
        records
    ):
        raise ValueError("practice audit does not cover every Gold race exactly once")
    accepted: dict[str, dict[str, Any]] = {}
    reasons = {}
    for record in records:
        key = _event(record["event_id"])
        if key not in events:
            raise ValueError("practice audit includes a non-Gold event")
        cutoff = events[key][0]["prediction_timestamp"]
        if record["cutoff"] != cutoff.isoformat():
            raise ValueError("practice audit cutoff differs from the source benchmark")
        if record["status"] != "audited":
            reasons[record["event_id"]] = record["reason"]
            continue
        document = _safe_file(root, record["document"]["path"], record["document"]["sha256"])
        registry = _safe_file(root, record["registry"]["path"], record["registry"]["sha256"])
        matching = [
            row
            for row in registry_rows(registry.read_text(encoding="utf-8"))
            if row["url"] == record["url"]
            and row["document_id"] in {None, record["document_id"]}
            and row["publication_cet"] == record["publication_cet"]
            and not row["recalled"]
        ]
        if len(matching) != 1:
            raise ValueError("practice audit lacks an exact retained FIA registry row")
        available = _publication(matching[0]) + timedelta(minutes=1)
        if available.isoformat() != record["available_at"]:
            raise ValueError("practice publication bound differs from the FIA registry")
        require_known_by(available, cutoff)
        parsed, field_size = parse_practice_pdf(document, {row["driver_id"] for row in events[key]})
        if parsed != record["practice"] or field_size != record["field_size"]:
            raise ValueError("practice audit values differ from the exact retained PDF")
        accepted[record["event_id"]] = {**record, "available_at": available}
    return accepted, reasons, digest


def build_gold_enrichment(
    root: Path,
    benchmark_dir: Path | None = None,
    catalog_path: Path | None = None,
    grid_audit_path: Path | None = None,
    practice_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze conservative prior-constructor features as a new benchmark version."""
    root = root.resolve()
    with advisory_lock(root / "data/features/gold_enrichment_v1/.build.lock"):
        benchmark = (
            benchmark_dir
            or root
            / "data/benchmarks/gold_core_rolling_v1"
            / json.loads(
                (root / "data/benchmarks/gold_core/manifest.json").read_text(encoding="utf-8")
            )["datasets"]["Gold"]["sha256"]
        )
        catalog = catalog_path or root / "data/benchmarks/gold_core_registry.json"
        manifest, rows, outcomes = _source(root, benchmark, catalog)
        events: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["cutoff_kind"] != "post_qualifying":
                raise ValueError("historical enrichment requires one post-qualifying cohort")
            events[_event(row["event_id"])].append(row)
        grids, grid_reasons, grid_audit_sha256 = (
            _grids(root, grid_audit_path, manifest["source_gold_sha256"], events)
            if grid_audit_path is not None
            else ({}, {}, "")
        )
        practices, practice_reasons, practice_audit_sha256 = (
            _practices(root, practice_audit_path, manifest["source_gold_sha256"], events)
            if practice_audit_path is not None
            else ({}, {}, "")
        )
        enriched, provenance = [], []
        for _key, current_rows in sorted(events.items()):
            cutoffs = {row["prediction_timestamp"] for row in current_rows}
            if len(cutoffs) != 1 or len({row["driver_id"] for row in current_rows}) != len(
                current_rows
            ):
                raise ValueError("Gold event has inconsistent cutoff or duplicate drivers")
            for row in sorted(current_rows, key=lambda item: item["driver_id"]):
                result, proof = _derive(
                    row,
                    current_rows,
                    events,
                    outcomes,
                    grids.get(row["event_id"]),
                    practices.get(row["event_id"]),
                )
                enriched.append(result)
                provenance.append(
                    {"event_id": row["event_id"], "driver_id": row["driver_id"], **proof}
                )
        source_gold = manifest["datasets"]["Gold"]["sha256"]
        output = root / "data/benchmarks/gold_historical_enrichment_v2" / source_gold
        if grid_audit_sha256:
            output = output / grid_audit_sha256
        if practice_audit_sha256:
            output = output / practice_audit_sha256
        output.mkdir(parents=True, exist_ok=True)
        source_schema = pq.read_schema(benchmark / "gold.parquet")
        source_fields = [
            pa.field("grid_position", pa.float64())
            if field.name == "grid_position"
            else pa.field("pit_lane_start", pa.bool_())
            if field.name == "pit_lane_start"
            else field
            for field in source_schema
        ]
        schema = pa.schema(
            [
                *source_fields,
                *(pa.field(name, pa.float64()) for name in ENRICHMENT_NUMERIC_FEATURES),
                *(
                    pa.field(f"{name}_missing", pa.bool_(), nullable=False)
                    for name in ENRICHMENT_NUMERIC_FEATURES
                ),
                *(pa.field(name, pa.int32(), nullable=False) for name in _COUNTS),
            ]
        )
        gold_path = output / "gold.parquet"
        if gold_path.exists():
            if pq.read_table(gold_path).to_pylist() != enriched:
                raise ValueError("immutable enriched Gold dataset collision")
        else:
            pq.write_table(
                pa.Table.from_pylist(enriched, schema=schema), gold_path, compression="zstd"
            )
        datasets = dict(manifest["datasets"])
        datasets["Gold"] = {**datasets["Gold"], "sha256": file_sha256(gold_path)}
        for tier in ("Silver", "Development"):
            original = benchmark / datasets[tier]["path"]
            replica = output / datasets[tier]["path"]
            if not replica.exists():
                shutil.copyfile(original, replica)
            if file_sha256(replica) != datasets[tier]["sha256"]:
                raise ValueError("enriched secondary tier bytes differ from source")
        coverage = output / "coverage.json"
        if not coverage.exists():
            shutil.copyfile(benchmark / "coverage.json", coverage)
        if file_sha256(coverage) != manifest["coverage_sha256"]:
            raise ValueError("source coverage changed")
        provenance_path = output / "feature_provenance.json"
        proof_bytes = _canonical(provenance)
        if provenance_path.exists() and provenance_path.read_bytes() != proof_bytes:
            raise ValueError("immutable enrichment provenance collision")
        if not provenance_path.exists():
            provenance_path.write_bytes(proof_bytes)
        new_manifest = {
            **manifest,
            "version": 3,
            "enrichment_version": ENRICHMENT_VERSION,
            "source_manifest_sha256": file_sha256(benchmark / "manifest.json"),
            "source_gold_sha256": source_gold,
            "feature_provenance_sha256": hashlib.sha256(proof_bytes).hexdigest(),
            "grid_audit_sha256": grid_audit_sha256 or None,
            "practice_audit_sha256": practice_audit_sha256 or None,
            "feature_columns": [*manifest["feature_columns"], *ENRICHMENT_FEATURE_COLUMNS],
            "datasets": datasets,
        }
        manifest_path = output / "manifest.json"
        new_bytes = _canonical(new_manifest)
        if manifest_path.exists() and manifest_path.read_bytes() != new_bytes:
            raise ValueError("immutable enriched manifest collision")
        if not manifest_path.exists():
            manifest_path.write_bytes(new_bytes)
        by_season: dict[str, dict[str, Any]] = {}
        for season in sorted({key[0] for key in events}):
            annual = [row for row in enriched if _event(row["event_id"])[0] == season]
            by_season[str(season)] = {
                "races": len({row["event_id"] for row in annual}),
                "rows": len(annual),
                "feature_nonmissing_rows": {
                    name: sum(row[name] is not None for row in annual)
                    for name in (*ENRICHMENT_NUMERIC_FEATURES, "grid_position")
                },
            }
        missing_reasons: dict[str, dict[str, int]] = {}
        for name in ENRICHMENT_NUMERIC_FEATURES:
            missing_reasons[name] = dict(
                sorted(
                    Counter(
                        record["reasons"][name]
                        for record in provenance
                        if name in record["reasons"]
                    ).items()
                )
            )
        report = {
            "version": ENRICHMENT_VERSION,
            "benchmark_dir": output.relative_to(root).as_posix(),
            "source_gold_sha256": source_gold,
            "enriched_gold_sha256": datasets["Gold"]["sha256"],
            "rows": len(enriched),
            "races": len(events),
            "coverage_by_season": by_season,
            "missing_reasons": missing_reasons,
            "evidence_sources": [
                "hash-verified Gold race outcomes and contemporary Gold constructor/qualifying rows"
            ],
            "point_ledger_status": "absent; no championship or constructor points certified",
            "grid_audit_sha256": grid_audit_sha256 or None,
            "grid_audited_races": len(grids),
            "practice_audited_races": len(practices),
            "grid_published_after_cutoff_races": sorted(
                event
                for event, reason in grid_reasons.items()
                if reason == "grid_published_after_post_qualifying_cutoff"
            ),
            "grid_missing_reasons": dict(sorted(Counter(grid_reasons.values()).items())),
            "practice_missing_race_reasons": dict(
                sorted(Counter(practice_reasons.values()).items())
            ),
        }
        report_path = output / "enrichment_report.json"
        report_bytes = _canonical(report)
        if report_path.exists() and report_path.read_bytes() != report_bytes:
            raise ValueError("immutable enrichment report collision")
        if not report_path.exists():
            report_path.write_bytes(report_bytes)
        return report
