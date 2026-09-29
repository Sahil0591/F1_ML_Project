"""Versioned rolling features derived only from earlier audited Gold races."""

import hashlib
import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import BENCHMARK_FEATURE_COLUMNS, _safe_file, file_sha256
from f1_ml_predictor.time import require_known_by, require_utc
from f1_ml_predictor.trust.locking import advisory_lock

ROLLING_VERSION = "gold-rolling-v1"
WINDOWS = (3, 5, 10)
ROLLING_FEATURE_COLUMNS = (
    *(f"recent_finish_mean_{window}" for window in WINDOWS),
    *(f"recent_dnf_rate_{window}" for window in WINDOWS),
    *(f"recent_finish_mean_{window}_missing" for window in WINDOWS),
    *(f"recent_dnf_rate_{window}_missing" for window in WINDOWS),
    *(f"history_count_{window}" for window in WINDOWS),
)
ROLLING_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("driver_id", pa.string(), nullable=False),
        pa.field("prediction_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        *[pa.field(f"recent_finish_mean_{window}", pa.float64()) for window in WINDOWS],
        *[pa.field(f"recent_dnf_rate_{window}", pa.float64()) for window in WINDOWS],
        *[pa.field(f"history_count_{window}", pa.int32(), nullable=False) for window in WINDOWS],
        pa.field("provenance", pa.string(), nullable=False),
    ]
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _event(event_id: str) -> tuple[int, int]:
    parts = event_id.split("/")
    if len(parts) != 2 or not parts[0].startswith("season=") or not parts[1].startswith("round="):
        raise ValueError("rolling input has an invalid event identity")
    return int(parts[0][7:]), int(parts[1][6:])


def build_gold_rolling(root: Path) -> dict[str, Any]:
    """Freeze within-season, contiguous prior-race windows from exact Gold labels.

    A missing prior round or a label unavailable at cutoff invalidates its entire
    window. A driver's absence from an audited race does not invent a result.
    """
    root = root.resolve()
    lock = root / "data/features/gold_rolling_v1/.build.lock"
    with advisory_lock(lock):
        return _build_gold_rolling_locked(root)


def _build_gold_rolling_locked(root: Path) -> dict[str, Any]:
    benchmark = root / "data/benchmarks/gold_core"
    manifest_path = benchmark / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry_path = root / "data/benchmarks/gold_core_registry.json"
    if file_sha256(registry_path) != manifest["catalog_sha256"]:
        raise ValueError("Gold registry changed since the benchmark was built")
    source = benchmark / manifest["datasets"]["Gold"]["path"]
    if file_sha256(source) != manifest["datasets"]["Gold"]["sha256"]:
        raise ValueError("Gold benchmark bytes differ from their manifest")
    source_rows = pq.read_table(source).to_pylist()
    if len(source_rows) != manifest["datasets"]["Gold"]["rows"]:
        raise ValueError("Gold benchmark row count changed")
    events: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in source_rows:
        if row["benchmark_tier"] != "Gold" or row["cutoff_kind"] != "post_qualifying":
            raise ValueError("rolling features require a post-qualifying Gold cohort")
        events.setdefault(_event(row["event_id"]), []).append(row)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    outcomes: dict[tuple[int, int], str] = {}
    for item in registry["races"]:
        key = _event(item["event_id"])
        if key in outcomes or key not in events:
            raise ValueError("Gold registry and benchmark event identities disagree")
        spec = item["outcomes"]
        _safe_file(root, spec["path"], spec["sha256"])
        outcomes[key] = spec["sha256"]
    if set(outcomes) != set(events):
        raise ValueError("Gold registry and benchmark coverage disagree")
    result = []
    for (season, round_number), rows in sorted(events.items()):
        for row in sorted(rows, key=lambda item: item["driver_id"]):
            cutoff = row["prediction_timestamp"]
            require_utc(cutoff, "rolling cutoff")
            features: dict[str, Any] = {
                "event_id": row["event_id"],
                "driver_id": row["driver_id"],
                "prediction_timestamp": cutoff,
            }
            proofs: dict[str, Any] = {}
            for window in WINDOWS:
                prior = [
                    (season, previous) for previous in range(round_number - window, round_number)
                ]
                complete = all(key[1] > 0 and key in events for key in prior)
                historical = [historic for key in prior for historic in events.get(key, [])]
                available = complete and all(
                    historic["label_final_audited"] is True
                    and historic["label_audit_reference"]
                    and historic["label_available_at"] is not None
                    and historic["label_available_at"] < cutoff
                    for historic in historical
                )
                if available:
                    for historic in historical:
                        require_known_by(historic["label_available_at"], cutoff)
                driver_rows = (
                    [
                        historic
                        for historic in historical
                        if historic["driver_id"] == row["driver_id"]
                    ]
                    if available
                    else []
                )
                positions = [
                    float(item["label_position"])
                    for item in driver_rows
                    if item["label_position"] is not None
                ]
                dnf = [item["label_dnf"] for item in driver_rows]
                features[f"recent_finish_mean_{window}"] = mean(positions) if positions else None
                features[f"recent_dnf_rate_{window}"] = (
                    mean(float(value) for value in dnf)
                    if dnf and all(isinstance(value, bool) for value in dnf)
                    else None
                )
                features[f"history_count_{window}"] = len(driver_rows)
                proofs[str(window)] = {
                    "complete": bool(available),
                    "source_events": [
                        {
                            "event_id": f"season={key[0]}/round={key[1]:02d}",
                            "outcome_sha256": outcomes[key],
                        }
                        for key in prior
                        if key in events
                    ],
                    "driver_observations": len(driver_rows),
                }
            features["provenance"] = _canonical(proofs).decode()
            result.append(features)
    table = pa.Table.from_pylist(result, schema=ROLLING_SCHEMA)
    directory = root / "data/features/gold_rolling_v1"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ".rolling-pending.parquet"
    pq.write_table(table, temporary, compression="zstd")
    digest = file_sha256(temporary)
    destination = directory / f"{digest}.parquet"
    if destination.exists():
        if file_sha256(destination) != digest:
            raise ValueError("immutable rolling feature collision")
        temporary.unlink()
    else:
        temporary.replace(destination)
    report = {
        "version": ROLLING_VERSION,
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_registry_sha256": file_sha256(registry_path),
        "source_gold_sha256": file_sha256(source),
        "feature_path": destination.relative_to(root).as_posix(),
        "feature_sha256": digest,
        "driver_race_observations": len(result),
        "unique_drivers": len({row["driver_id"] for row in result}),
        "race_count": len(events),
        "windows": list(WINDOWS),
        "policy": "same-season contiguous audited prior races; all labels available before cutoff",
    }
    output = root / "data/benchmarks/gold_core_rolling_v1" / str(report["source_gold_sha256"])
    output.mkdir(parents=True, exist_ok=True)
    rolling_by_key = {
        (row["event_id"], row["driver_id"], row["prediction_timestamp"]): row for row in result
    }
    enriched = []
    for row in source_rows:
        row_key = row["event_id"], row["driver_id"], row["prediction_timestamp"]
        rolling = rolling_by_key[row_key]
        additions = {
            name: rolling[name] for name in ROLLING_FEATURE_COLUMNS if not name.endswith("_missing")
        }
        for name in ROLLING_FEATURE_COLUMNS:
            if name.endswith("_missing"):
                additions[name] = rolling[name.removesuffix("_missing")] is None
        enriched.append({**row, **additions})
    enriched_path = output / "gold.parquet"
    if enriched_path.exists():
        if pq.read_table(enriched_path).to_pylist() != enriched:
            raise ValueError("immutable rolling benchmark collision")
    else:
        pq.write_table(pa.Table.from_pylist(enriched), enriched_path, compression="zstd")
    datasets = dict(manifest["datasets"])
    datasets["Gold"] = {
        **datasets["Gold"],
        "sha256": file_sha256(enriched_path),
    }
    for tier in ("Silver", "Development"):
        original = benchmark / str(datasets[tier]["path"])
        replica = output / str(datasets[tier]["path"])
        if not replica.exists():
            shutil.copyfile(original, replica)
        if file_sha256(replica) != datasets[tier]["sha256"]:
            raise ValueError("rolling benchmark secondary tier bytes changed")
    coverage = output / "coverage.json"
    if not coverage.exists():
        shutil.copyfile(benchmark / "coverage.json", coverage)
    if file_sha256(coverage) != manifest["coverage_sha256"]:
        raise ValueError("rolling benchmark coverage changed")
    new_manifest = {
        **manifest,
        "version": 2,
        "rolling_version": ROLLING_VERSION,
        "source_gold_sha256": report["source_gold_sha256"],
        "rolling_feature_sha256": digest,
        "feature_columns": [*BENCHMARK_FEATURE_COLUMNS, *ROLLING_FEATURE_COLUMNS],
        "datasets": datasets,
    }
    enriched_manifest = output / "manifest.json"
    if enriched_manifest.exists() and enriched_manifest.read_bytes() != _canonical(new_manifest):
        raise ValueError("immutable rolling benchmark manifest collision")
    if not enriched_manifest.exists():
        with enriched_manifest.open("xb") as handle:
            handle.write(_canonical(new_manifest))
    report["benchmark_dir"] = output.relative_to(root).as_posix()
    report["benchmark_gold_sha256"] = datasets["Gold"]["sha256"]
    report_path = directory / f"{hashlib.sha256(_canonical(report)).hexdigest()}.json"
    if report_path.exists() and report_path.read_bytes() != _canonical(report):
        raise ValueError("immutable rolling manifest collision")
    if not report_path.exists():
        with report_path.open("xb") as handle:
            handle.write(_canonical(report))
    return report
