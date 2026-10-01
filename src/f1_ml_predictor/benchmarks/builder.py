"""Build deterministic race benchmarks from verified snapshots and audited labels."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.features.snapshot import (
    CONSERVATIVE_ALIASES,
    FEATURE_SCHEMA,
    LEGACY_FEATURE_SCHEMA,
    NUMERIC_FEATURES,
)
from f1_ml_predictor.identifiers import EntityId, EntityKind, EventId
from f1_ml_predictor.time import require_known_by, require_utc
from f1_ml_predictor.trust.evidence import BenchmarkTier, EvidenceClass, evidence_from_dict
from f1_ml_predictor.trust.outcomes import OUTCOME_SCHEMA, validate_audited_outcomes

_TIERS = (BenchmarkTier.GOLD, BenchmarkTier.SILVER, BenchmarkTier.DEVELOPMENT)
_LABEL_COLUMNS = tuple(
    field.name for field in OUTCOME_SCHEMA if field.name not in {"season", "round", "driver_id"}
)
_LEGACY_FOR_ALIAS = CONSERVATIVE_ALIASES
_BENCHMARK_FEATURES = tuple(
    next((modern for modern, legacy in _LEGACY_FOR_ALIAS.items() if legacy == name), name)
    for name in NUMERIC_FEATURES
)
BENCHMARK_FEATURE_COLUMNS = (
    *_BENCHMARK_FEATURES,
    *(f"{name}_missing" for name in _BENCHMARK_FEATURES),
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _label_column(name: str) -> str:
    return name if name.startswith("label_") else f"label_{name}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(root: Path, relative: str, expected_hash: str) -> Path:
    if not isinstance(relative, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("catalog paths require a relative path and SHA-256")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("catalog paths must stay within the workspace")
    path = (root / candidate).resolve()
    base = root.resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.is_symlink():
        raise ValueError("catalog input is missing or escapes the workspace")
    if file_sha256(path) != expected_hash:
        raise ValueError("catalog input hash does not match the file")
    return path


def _tier_from_evidence(row: dict[str, Any]) -> BenchmarkTier:
    raw = row.get("feature_evidence")
    declared = row.get("benchmark_tier", BenchmarkTier.DEVELOPMENT.value)
    try:
        declared_tier = BenchmarkTier(declared)
    except ValueError as exc:
        raise ValueError("feature row has an unsupported benchmark tier") from exc
    if raw is None:
        return BenchmarkTier.DEVELOPMENT
    evidence = json.loads(raw)
    tiers: list[BenchmarkTier] = []
    if all(row.get(feature) is None for feature in NUMERIC_FEATURES):
        context = evidence.get("__context__")
        if context is not None:
            tiers.append(_context_tier(row, context))
    for feature in NUMERIC_FEATURES:
        if row.get(feature) is None:
            continue
        item = evidence.get(feature)
        if not isinstance(item, dict) or item.get("missing") is not False:
            raise ValueError(f"feature evidence is missing for {feature}")
        if not isinstance(item.get("inputs"), list) or not item["inputs"]:
            raise ValueError(f"feature provenance is missing for {feature}")
        input_tiers = []
        for source in item["inputs"]:
            proof = source.get("evidence")
            if not isinstance(proof, dict):
                raise ValueError("feature input evidence is malformed")
            kind = EvidenceClass(proof["class"])
            _validate_evidence_record(proof)
            available = proof.get("available_at")
            if isinstance(available, str):
                require_known_by(datetime.fromisoformat(available), row["prediction_timestamp"])
            audited = proof.get("audited") is True
            if kind == EvidenceClass.CAPTURED_LIVE:
                input_tiers.append(BenchmarkTier.GOLD)
            elif kind == EvidenceClass.CONSERVATIVE_RECONSTRUCTION and audited:
                input_tiers.append(BenchmarkTier.SILVER)
            elif (
                kind in {EvidenceClass.SOURCE_PUBLISHED_TIMESTAMP, EvidenceClass.VERSIONED_ARCHIVE}
                and audited
            ):
                input_tiers.append(BenchmarkTier.GOLD)
            else:
                input_tiers.append(BenchmarkTier.DEVELOPMENT)
        weakest = max(input_tiers, key=_TIERS.index)
        if item.get("tier") != declared_tier.value and declared_tier != BenchmarkTier.DEVELOPMENT:
            raise ValueError("feature and row evidence tiers disagree")
        tiers.append(weakest)
    derived = max(tiers, key=_TIERS.index) if tiers else BenchmarkTier.DEVELOPMENT
    if _TIERS.index(declared_tier) < _TIERS.index(derived):
        raise ValueError("feature row claims a stronger tier than its evidence")
    return declared_tier


def _context_tier(row: dict[str, Any], context: Any) -> BenchmarkTier:
    """Validate required published context without inventing an absent numeric value."""
    if not isinstance(context, dict) or context.get("tier") != row["benchmark_tier"]:
        raise ValueError("missing-value context has an inconsistent tier")
    provenance = json.loads(row["provenance"])
    records = provenance.get("inputs")
    sources = context.get("inputs")
    if (
        provenance.get("event_id") != row["event_id"]
        or not isinstance(records, list)
        or len(records) < 2
        or not isinstance(sources, list)
        or len(sources) != len(records) + 1
    ):
        raise ValueError("missing-value context lacks required event, roster and qualifying inputs")
    event_reference = provenance["event_reference"]
    event_proof = provenance.get("event_evidence")
    if not isinstance(event_proof, dict):
        raise ValueError("missing-value context lacks event evidence")
    expected = {
        (record["reference"], record["sha256"], record["available_at"]) for record in records
    }
    event_key = (
        event_reference,
        event_proof.get("artifact_sha256"),
        event_proof.get("available_at"),
    )
    expected.add(event_key)
    seen = set()
    tiers = []
    for source in sources:
        reference = source.get("reference")
        proof = source.get("evidence")
        if not isinstance(proof, dict):
            raise ValueError("missing-value context has malformed evidence")
        key = (reference, proof.get("artifact_sha256"), proof.get("available_at"))
        if key in seen or proof.get("reference") != reference:
            raise ValueError("missing-value context has malformed or duplicate evidence")
        _validate_evidence_record(proof)
        seen.add(key)
        if key == event_key:
            if proof != event_proof:
                raise ValueError("missing-value event context does not match its provenance")
        elif key not in expected:
            raise ValueError("missing-value input context does not match its exact provenance")
        available = proof.get("available_at")
        if isinstance(available, str):
            require_known_by(datetime.fromisoformat(available), row["prediction_timestamp"])
        tiers.append(evidence_from_dict(proof).tier)
    if seen != expected:
        raise ValueError("missing-value context omits a required input")
    return max(tiers, key=_TIERS.index)


def _validate_evidence_record(proof: dict[str, Any]) -> None:
    evidence_from_dict(proof)
    if not isinstance(proof.get("reference"), str) or not proof["reference"].strip():
        raise ValueError("feature input evidence has no reference")
    available = proof.get("available_at")
    captured = proof.get("captured_at")
    kind = EvidenceClass(proof["class"])
    if kind != EvidenceClass.CURRENT_STATE_ONLY:
        artifact_hash = proof.get("artifact_sha256")
        if (
            not isinstance(available, str)
            or not isinstance(artifact_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", artifact_hash)
        ):
            raise ValueError("feature input evidence lacks a bounded exact artifact")
        assert isinstance(available, str)
        available_at = datetime.fromisoformat(available)
        require_utc(available_at, "evidence available_at")
    if isinstance(available, str):
        published_bound = datetime.fromisoformat(available)
        require_utc(published_bound, "evidence available_at")
    elif kind != EvidenceClass.CURRENT_STATE_ONLY:
        raise ValueError("temporal evidence has no availability bound")
    if kind == EvidenceClass.CAPTURED_LIVE:
        if captured != available:
            raise ValueError("captured-live input evidence has inconsistent capture time")
    elif captured is not None:
        if not isinstance(captured, str) or not isinstance(available, str):
            raise ValueError("evidence capture time must be a timestamp")
        captured_at = datetime.fromisoformat(captured)
        require_utc(captured_at, "evidence captured_at")
        require_known_by(
            datetime.fromisoformat(available),
            captured_at,
        )
    if isinstance(available, str) and "prediction_timestamp" in proof:
        require_known_by(datetime.fromisoformat(available), proof["prediction_timestamp"])
    if proof.get("source_published_at") is not None:
        published_at = proof["source_published_at"]
        if not isinstance(published_at, str) or not isinstance(available, str):
            raise ValueError("source publication evidence timestamps must be strings")
        published = datetime.fromisoformat(published_at)
        require_utc(published, "source_published_at")
        require_known_by(published, datetime.fromisoformat(available))
    if kind == EvidenceClass.VERSIONED_ARCHIVE and not proof.get("archive_version"):
        raise ValueError("versioned archive evidence lacks an archive version")
    if kind == EvidenceClass.CONSERVATIVE_RECONSTRUCTION and not proof.get("reconstruction_method"):
        raise ValueError("reconstruction evidence lacks its method")


def _load_catalog(path: Path) -> list[dict[str, Any]]:
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read benchmark catalog") from exc
    if not isinstance(catalog, dict) or catalog.get("version") != 1:
        raise ValueError("benchmark catalog must be version 1")
    races = catalog.get("races")
    if not isinstance(races, list):
        raise ValueError("benchmark catalog races must be a list")
    return races


def discover_local_races(root: Path) -> list[dict[str, Any]]:
    """Find local race partitions without asserting that their contents are certified."""
    races = []
    for entries in sorted((root / "data" / "normalized").glob("season=*/round=*/entries.parquet")):
        season = int(entries.parents[1].name.split("=")[1])
        round_number = int(entries.parent.name.split("=")[1])
        event = EventId(season, round_number)
        races.append(
            {
                "event_id": event.partition(),
                "features": None,
                "outcomes": None,
                "discovery": "normalized_entry_partition",
            }
        )
    return races


def _validate_feature_table(
    table: pa.Table, event: EventId, cutoff: Any, cutoff_kind: str
) -> tuple[list[dict[str, Any]], BenchmarkTier]:
    schema = table.schema.remove_metadata()
    if not any(schema.equals(candidate) for candidate in (FEATURE_SCHEMA, LEGACY_FEATURE_SCHEMA)):
        raise ValueError("feature snapshot has an unsupported schema")
    if table.num_rows == 0:
        raise ValueError("feature snapshot is empty")
    rows = table.to_pylist()
    if len({row["driver_id"] for row in rows}) != len(rows):
        raise ValueError("feature snapshot contains duplicate drivers")
    tiers = set()
    for row in rows:
        if row["event_id"] != event.partition() or row["prediction_timestamp"] != cutoff:
            raise ValueError("feature event/cutoff does not match the catalog")
        if "cutoff_kind" in row and row["cutoff_kind"] != cutoff_kind:
            raise ValueError("feature cutoff kind does not match the catalog")
        require_known_by(row["feature_timestamp"], cutoff)
        EntityId(EntityKind.DRIVER, row["driver_id"])
        EntityId(EntityKind.CONSTRUCTOR, row["constructor_id"])
        EntityId(EntityKind.CIRCUIT, row["circuit_id"])
        tiers.add(_tier_from_evidence(row))
    if len(tiers) != 1:
        raise ValueError("one race snapshot cannot contain multiple evidence tiers")
    return rows, next(iter(tiers))


def _read_candidate(
    root: Path, item: dict[str, Any]
) -> tuple[EventId, Any, str, pa.Table, pa.Table]:
    if not isinstance(item, dict):
        raise ValueError("benchmark race entries must be objects")
    event_text = item.get("event_id")
    if not isinstance(event_text, str):
        raise ValueError("benchmark race requires an event_id")
    match = re.fullmatch(r"season=([0-9]{4})/round=([0-9]{2})", event_text)
    if not match:
        raise ValueError("benchmark event_id is not canonical")
    event = EventId(int(match[1]), int(match[2]))
    if event.partition() != event_text:
        raise ValueError("benchmark event_id is not canonical")
    cutoff_raw = item.get("prediction_timestamp")
    if not isinstance(cutoff_raw, str):
        raise ValueError("benchmark race requires its prediction timestamp")
    cutoff = datetime.fromisoformat(cutoff_raw)
    require_utc(cutoff, "prediction_timestamp")
    cutoff_kind = item.get("cutoff_kind")
    if cutoff_kind not in {"post_qualifying", "provisional_grid", "pre_race"}:
        raise ValueError("benchmark race needs an explicit supported cutoff kind")
    features_spec, outcomes_spec = item.get("features"), item.get("outcomes")
    if not isinstance(features_spec, dict) or not isinstance(outcomes_spec, dict):
        raise ValueError("benchmark race needs feature and audited outcome file references")
    feature_relative = features_spec.get("path")
    feature_hash = features_spec.get("sha256")
    outcome_relative = outcomes_spec.get("path")
    outcome_hash = outcomes_spec.get("sha256")
    if not isinstance(feature_relative, str) or not isinstance(feature_hash, str):
        raise ValueError("feature input path and hash must be strings")
    if not isinstance(outcome_relative, str) or not isinstance(outcome_hash, str):
        raise ValueError("benchmark input paths and hashes must be strings")
    feature_path = _safe_file(root, feature_relative, feature_hash)
    outcome_path = _safe_file(root, outcome_relative, outcome_hash)
    return (
        event,
        cutoff,
        cutoff_kind,
        pq.ParquetFile(feature_path).read(),
        pq.ParquetFile(outcome_path).read(),
    )


def _empty_table() -> pa.Table:
    return pa.Table.from_pylist([], schema=_benchmark_schema())


def _benchmark_schema() -> pa.Schema:
    fields = [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("constructor_id", pa.string(), nullable=False),
        pa.field("circuit_id", pa.string(), nullable=False),
        pa.field("prediction_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("feature_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("benchmark_tier", pa.string(), nullable=False),
        pa.field("cutoff_kind", pa.string(), nullable=False),
        pa.field("qualifying_status", pa.string(), nullable=False),
        pa.field("start_type", pa.string(), nullable=False),
        pa.field("pit_lane_start", pa.bool_()),
        pa.field("grid_status", pa.string(), nullable=False),
        *[pa.field(name, pa.float64()) for name in _BENCHMARK_FEATURES],
        *[pa.field(f"{name}_missing", pa.bool_(), nullable=False) for name in _BENCHMARK_FEATURES],
        *[
            pa.field(_label_column(name), field.type, nullable=field.nullable)
            for field in OUTCOME_SCHEMA
            if field.name in _LABEL_COLUMNS
            for name in (field.name,)
        ],
    ]
    return pa.schema(fields)


def _assemble_rows(
    event: EventId,
    cutoff: Any,
    cutoff_kind: str,
    features: list[dict[str, Any]],
    outcomes: pa.Table,
    tier: BenchmarkTier,
) -> list[dict[str, Any]]:
    if not outcomes.schema.remove_metadata().equals(OUTCOME_SCHEMA):
        raise ValueError("outcomes do not match the final audited outcome schema")
    roster = {(event, row["driver_id"]) for row in features}
    validate_audited_outcomes(outcomes, field_roster=roster)
    labels = {row["driver_id"]: row for row in outcomes.to_pylist()}
    if sum(bool(row["winner"]) for row in labels.values()) != 1:
        raise ValueError("each complete race benchmark must have exactly one winner")
    for label in labels.values():
        if label["label_available_at"] <= cutoff:
            raise ValueError("final audited labels must be published after the prediction cutoff")
    result = []
    for feature in features:
        row = {
            "event_id": event.partition(),
            "driver_id": feature["driver_id"],
            "constructor_id": feature["constructor_id"],
            "circuit_id": feature["circuit_id"],
            "prediction_timestamp": cutoff,
            "feature_timestamp": feature["feature_timestamp"],
            "benchmark_tier": tier.value,
            "cutoff_kind": cutoff_kind,
            "qualifying_status": feature.get("qualifying_status", "unknown"),
            "start_type": feature.get("start_type", "unknown"),
            "pit_lane_start": feature.get("pit_lane_start"),
            "grid_status": feature.get("grid_status", "unknown"),
        }
        for name in _BENCHMARK_FEATURES:
            legacy = _LEGACY_FOR_ALIAS.get(name, name)
            source_name = name if name in feature else legacy
            row[name] = feature.get(source_name)
            missing_name = f"{name}_missing"
            legacy_missing = f"{legacy}_missing"
            row[missing_name] = feature.get(
                missing_name, feature.get(legacy_missing, row[name] is None)
            )
        row.update(
            {_label_column(name): labels[feature["driver_id"]][name] for name in _LABEL_COLUMNS}
        )
        result.append(row)
    return result


def build_benchmarks(root: Path, output: Path, catalog_path: Path | None = None) -> dict[str, Any]:
    """Write tier Parquet files, an inclusion manifest and race coverage report."""
    from f1_ml_predictor.benchmarks.versioning import archive_benchmark

    archive_benchmark(output)
    races = _load_catalog(catalog_path) if catalog_path else discover_local_races(root)
    outputs: dict[BenchmarkTier, list[dict[str, Any]]] = {tier: [] for tier in _TIERS}
    coverage = []
    seen: set[tuple[str, str, str]] = set()
    for item in races:
        event_text = item.get("event_id") if isinstance(item, dict) else None
        reasons = []
        missing_features = list(_BENCHMARK_FEATURES)
        evidence_quality = "unknown"
        if not isinstance(event_text, str):
            coverage.append(
                {
                    "event_id": event_text,
                    "status": "excluded",
                    "reasons": ["invalid_or_duplicate_event_id"],
                }
            )
            continue
        cohort = (
            event_text,
            str(item.get("prediction_timestamp")) if isinstance(item, dict) else "",
            str(item.get("cutoff_kind")) if isinstance(item, dict) else "",
        )
        source_files = {}
        if isinstance(item, dict):
            for name in ("features", "outcomes"):
                reference = item.get(name)
                if isinstance(reference, dict):
                    source_files[name] = {key: reference.get(key) for key in ("path", "sha256")}
        if cohort in seen:
            coverage.append(
                {
                    "event_id": event_text,
                    "status": "excluded",
                    "reasons": ["invalid_or_duplicate_event_id"],
                }
            )
            continue
        seen.add(cohort)
        if item.get("features") is None:
            reasons.append("feature_snapshot_not_registered")
        if item.get("outcomes") is None:
            reasons.append("audited_outcomes_not_registered")
        if reasons:
            coverage.append(
                {
                    "event_id": event_text,
                    "cutoff_kind": item.get("cutoff_kind"),
                    "source_files": source_files,
                    "status": "excluded",
                    "evidence_quality": evidence_quality,
                    "missing_features": missing_features,
                    "reasons": reasons,
                }
            )
            continue
        try:
            event, cutoff, cutoff_kind, feature_table, outcome_table = _read_candidate(root, item)
            feature_rows, tier = _validate_feature_table(feature_table, event, cutoff, cutoff_kind)
            evidence_quality = tier.value
            missing_features = [
                name
                for name in _BENCHMARK_FEATURES
                if any(row.get(name) is None for row in feature_rows)
            ]
            missing_feature_counts = {
                name: sum(row.get(name) is None for row in feature_rows)
                for name in _BENCHMARK_FEATURES
                if any(row.get(name) is None for row in feature_rows)
            }
            combined = _assemble_rows(event, cutoff, cutoff_kind, feature_rows, outcome_table, tier)
            outputs[tier].extend(combined)
            coverage.append(
                {
                    "event_id": event_text,
                    "cutoff_kind": item.get("cutoff_kind"),
                    "source_files": source_files,
                    "status": "included",
                    "tier": tier.value,
                    "evidence_quality": tier.value,
                    "driver_rows": len(combined),
                    "missing_features": missing_features,
                    "missing_feature_counts": missing_feature_counts,
                    "reasons": [],
                }
            )
        except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
            reasons.append(str(exc))
            coverage.append(
                {
                    "event_id": event_text,
                    "cutoff_kind": item.get("cutoff_kind"),
                    "source_files": source_files,
                    "status": "excluded",
                    "evidence_quality": evidence_quality,
                    "missing_features": missing_features,
                    "reasons": reasons,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for tier in _TIERS:
        rows = sorted(outputs[tier], key=lambda row: (row["event_id"], row["driver_id"]))
        table = pa.Table.from_pylist(rows, schema=_benchmark_schema()) if rows else _empty_table()
        path = output / f"{tier.value.lower()}.parquet"
        pq.write_table(table, path, compression="zstd")
        datasets[tier.value] = {
            "path": path.name,
            "sha256": file_sha256(path),
            "rows": len(rows),
            "events": sorted({row["event_id"] for row in rows}),
        }
    report = {
        "version": 1,
        "datasets": datasets,
        "included_races": sum(row["status"] == "included" for row in coverage),
        "excluded_races": sum(row["status"] == "excluded" for row in coverage),
        "coverage": sorted(coverage, key=lambda row: str(row.get("event_id"))),
    }
    report_bytes = _canonical_json(report)
    (output / "coverage.json").write_bytes(report_bytes)
    manifest = {
        "version": 1,
        "catalog_source": "explicit" if catalog_path else "local_discovery",
        "catalog_sha256": file_sha256(catalog_path)
        if catalog_path
        else hashlib.sha256(_canonical_json(races)).hexdigest(),
        "feature_columns": list(BENCHMARK_FEATURE_COLUMNS),
        "context_columns": [
            "cutoff_kind",
            "qualifying_status",
            "start_type",
            "pit_lane_start",
            "grid_status",
        ],
        "label_columns": [_label_column(name) for name in _LABEL_COLUMNS],
        "primary_accuracy_tier": BenchmarkTier.GOLD.value,
        "coverage_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "datasets": datasets,
    }
    (output / "manifest.json").write_bytes(_canonical_json(manifest))
    archive_benchmark(output)
    return report
