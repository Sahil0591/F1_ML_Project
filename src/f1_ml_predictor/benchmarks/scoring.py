"""Create an immutable Gold scoring feature version from audited point evidence."""

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import _safe_file, file_sha256
from f1_ml_predictor.benchmarks.enrichment import ENRICHMENT_VERSION
from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.scoring.ledger import ScoringLedger, load_scoring_ledger
from f1_ml_predictor.time import require_known_by
from f1_ml_predictor.trust.locking import advisory_lock

SCORING_VERSION = "gold-championship-scoring-v1"
SCORING_FEATURES = (
    *(f"driver_points_last_{window}" for window in (3, 5, 10)),
    *(f"constructor_points_last_{window}" for window in (3, 5, 10)),
)
NEW_FEATURE_COLUMNS = (
    *(f"driver_points_last_{window}" for window in (3, 5, 10)),
    *(f"driver_points_last_{window}_missing" for window in (3, 5, 10)),
)
_POINT_FEATURES = (
    "driver_points_before_race",
    "driver_championship_position",
    "driver_points_gap_to_leader",
    "constructor_points_before_race",
    "constructor_championship_position",
    "constructor_points_gap_to_leader",
    *SCORING_FEATURES,
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def verify_scoring_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    """Reject an incomplete or altered version 4 scoring provenance bundle."""
    if manifest.get("version") != 4:
        return
    if manifest.get("scoring_version") != SCORING_VERSION:
        raise ValueError("unsupported scoring benchmark version")
    for name in (
        "scoring_ledger_sha256",
        "scoring_rules_sha256",
        "event_points_evidence_sha256",
        "scoring_provenance_sha256",
        "source_manifest_sha256",
    ):
        value = manifest.get(name)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"invalid scoring manifest {name}")
    if file_sha256(directory / "scoring_provenance.json") != manifest["scoring_provenance_sha256"]:
        raise ValueError("scoring feature provenance hash mismatch")


def _event(value: str) -> EventId:
    season, round_number = value.split("/")
    event = EventId(int(season.removeprefix("season=")), int(round_number.removeprefix("round=")))
    if event.partition() != value:
        raise ValueError("noncanonical Gold event identifier")
    return event


def _proof(ledger: ScoringLedger, event: EventId, cutoff: Any) -> list[dict[str, Any]]:
    proof = []
    for number in range(1, event.round):
        versions = [
            version
            for version in ledger.events
            if version.event == EventId(event.season, number)
            and version.completed_at < cutoff
            and version.effective_at < cutoff
        ]
        if not versions:
            break
        selected = max(versions, key=lambda version: version.effective_at)
        proof.append(
            {
                "event_id": selected.event.partition(),
                "effective_at": selected.effective_at.isoformat(),
                "race_schedule": selected.race_schedule,
                "evidence_hash": selected.evidence_hash,
                "awards": [
                    {
                        "driver_id": entry.driver_id,
                        "constructor_id": entry.constructor_id,
                        "race_points": entry.race_points,
                        "sprint_points": entry.sprint_points,
                        "bonus_points": entry.bonus_points,
                        "adjustment_points": entry.adjustment_points,
                        "total_points": entry.total_points,
                        "effective_at": entry.effective_at.isoformat(),
                        "source_evidence": entry.source_evidence,
                        "evidence_hash": entry.evidence_hash,
                    }
                    for entry in selected.entries
                ],
            }
        )
    return proof


def build_gold_scoring(
    root: Path,
    benchmark_dir: Path,
    rules_path: Path | None = None,
    evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze point features when audit files exist; preserve the source benchmark."""
    root = root.resolve()
    rules_path = rules_path or root / "data/audit/scoring_rules.json"
    evidence_path = evidence_path or root / "data/audit/event_points_evidence.json"
    ledger = load_scoring_ledger(rules_path, evidence_path)
    benchmark_dir = benchmark_dir.resolve()
    with advisory_lock(root / "data/benchmarks/gold_championship_scoring_v1/.build.lock"):
        manifest_path = benchmark_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 3 or manifest.get("enrichment_version") != ENRICHMENT_VERSION:
            raise ValueError("scoring requires the immutable historical Gold enrichment")
        if file_sha256(benchmark_dir / "coverage.json") != manifest["coverage_sha256"]:
            raise ValueError("source coverage hash mismatch")
        if (
            file_sha256(benchmark_dir / "feature_provenance.json")
            != manifest["feature_provenance_sha256"]
        ):
            raise ValueError("source feature provenance hash mismatch")
        for record in manifest["datasets"].values():
            _safe_file(benchmark_dir, record["path"], record["sha256"])
        source = pq.read_table(benchmark_dir / manifest["datasets"]["Gold"]["path"])
        reserved = [name for name in _POINT_FEATURES if not name.startswith("driver_points_last_")]
        if any(
            name not in source.column_names
            or not pa.types.is_float64(source.schema.field(name).type)
            or f"{name}_missing" not in source.column_names
            or not pa.types.is_boolean(source.schema.field(f"{name}_missing").type)
            for name in reserved
        ):
            raise ValueError("source Gold has an incomplete scoring feature schema")
        rows = source.to_pylist()
        if len(rows) != manifest["datasets"]["Gold"]["rows"]:
            raise ValueError("source Gold row count mismatch")
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["benchmark_tier"] != "Gold" or row["cutoff_kind"] != "post_qualifying":
                raise ValueError("scoring source must contain post-qualifying Gold only")
            groups[row["event_id"]].append(row)
        if set(groups) != set(manifest["datasets"]["Gold"]["events"]):
            raise ValueError("source Gold event set mismatch")
        enriched = []
        provenance = []
        for event_id, event_rows in sorted(groups.items()):
            event = _event(event_id)
            cutoffs = {row["prediction_timestamp"] for row in event_rows}
            drivers = {row["driver_id"]: row["constructor_id"] for row in event_rows}
            if len(cutoffs) != 1 or len(drivers) != len(event_rows):
                raise ValueError("inconsistent Gold event roster or cutoff")
            cutoff = next(iter(cutoffs))
            features = ledger.standings_before(event, cutoff, drivers)
            history = _proof(ledger, event, cutoff)
            for row in sorted(event_rows, key=lambda value: value["driver_id"]):
                values = features[row["driver_id"]]
                updated = {**row}
                if values["missing_reason"] is None and history:
                    updated["feature_timestamp"] = max(
                        row["feature_timestamp"],
                        *(datetime.fromisoformat(record["effective_at"]) for record in history),
                    )
                require_known_by(updated["feature_timestamp"], cutoff)
                missing = {}
                for name in _POINT_FEATURES:
                    value = values[name]
                    updated[name] = value
                    updated[f"{name}_missing"] = value is None
                    if value is None:
                        missing[name] = (
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
                enriched.append(updated)
                provenance.append(
                    {
                        "event_id": event_id,
                        "driver_id": row["driver_id"],
                        "constructor_id": row["constructor_id"],
                        "cutoff": cutoff.isoformat(),
                        "feature_available_at": updated["feature_timestamp"].isoformat(),
                        "scoring_ledger_sha256": ledger.sha256,
                        "source_event_versions": history,
                        "missing_reasons": missing,
                    }
                )
        source_hash = file_sha256(manifest_path)
        output = root / "data/benchmarks/gold_championship_scoring_v1" / source_hash / ledger.sha256
        output.mkdir(parents=True, exist_ok=True)
        new_fields = [
            *(pa.field(name, pa.float64()) for name in NEW_FEATURE_COLUMNS[:3]),
            *(pa.field(name, pa.bool_(), nullable=False) for name in NEW_FEATURE_COLUMNS[3:]),
        ]
        gold_path = output / "gold.parquet"
        if gold_path.exists():
            if pq.read_table(gold_path).to_pylist() != enriched:
                raise ValueError("immutable scoring Gold collision")
        else:
            pq.write_table(
                pa.Table.from_pylist(enriched, schema=pa.schema([*source.schema, *new_fields])),
                gold_path,
                compression="zstd",
            )
        datasets = dict(manifest["datasets"])
        datasets["Gold"] = {**datasets["Gold"], "sha256": file_sha256(gold_path)}
        for tier in ("Silver", "Development"):
            source_file = benchmark_dir / datasets[tier]["path"]
            target = output / datasets[tier]["path"]
            if not target.exists():
                shutil.copyfile(source_file, target)
            if file_sha256(target) != datasets[tier]["sha256"]:
                raise ValueError("secondary tier changed during scoring build")
        coverage = output / "coverage.json"
        if not coverage.exists():
            shutil.copyfile(benchmark_dir / "coverage.json", coverage)
        proof_path = output / "scoring_provenance.json"
        proof_bytes = _canonical(provenance)
        if proof_path.exists() and proof_path.read_bytes() != proof_bytes:
            raise ValueError("immutable scoring provenance collision")
        if not proof_path.exists():
            proof_path.write_bytes(proof_bytes)
        new_manifest = {
            **manifest,
            "version": 4,
            "scoring_version": SCORING_VERSION,
            "scoring_ledger_sha256": ledger.sha256,
            "scoring_rules_sha256": ledger.rules_sha256,
            "event_points_evidence_sha256": ledger.evidence_sha256,
            "scoring_provenance_sha256": hashlib.sha256(proof_bytes).hexdigest(),
            "source_manifest_sha256": source_hash,
            "feature_columns": [*manifest["feature_columns"], *NEW_FEATURE_COLUMNS],
            "datasets": datasets,
        }
        new_manifest_path = output / "manifest.json"
        manifest_bytes = _canonical(new_manifest)
        if new_manifest_path.exists() and new_manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("immutable scoring manifest collision")
        if not new_manifest_path.exists():
            new_manifest_path.write_bytes(manifest_bytes)
        by_season = {}
        for season in sorted({_event(event_id).season for event_id in groups}):
            annual = [row for row in enriched if _event(row["event_id"]).season == season]
            by_season[str(season)] = {
                "races": len({row["event_id"] for row in annual}),
                "rows": len(annual),
                "nonmissing_rows": {
                    name: sum(row[name] is not None for row in annual) for name in _POINT_FEATURES
                },
            }
        report = {
            "version": SCORING_VERSION,
            "benchmark_dir": output.relative_to(root).as_posix(),
            "scoring_ledger_sha256": ledger.sha256,
            "coverage_by_season": by_season,
            "missing_reasons": {
                name: dict(
                    sorted(
                        Counter(
                            proof["missing_reasons"][name]
                            for proof in provenance
                            if name in proof["missing_reasons"]
                        ).items()
                    )
                )
                for name in _POINT_FEATURES
            },
        }
        report_path = output / "scoring_report.json"
        report_bytes = _canonical(report)
        if report_path.exists() and report_path.read_bytes() != report_bytes:
            raise ValueError("immutable scoring report collision")
        if not report_path.exists():
            report_path.write_bytes(report_bytes)
        return report
