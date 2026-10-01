"""Immutable benchmark snapshots and race eligibility history."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.trust.locking import advisory_lock


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _verified_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if file_sha256(directory / "coverage.json") != manifest["coverage_sha256"]:
        raise ValueError("benchmark coverage hash mismatch")
    for record in manifest["datasets"].values():
        name = record["path"]
        if Path(name).name != name or file_sha256(directory / name) != record["sha256"]:
            raise ValueError("benchmark dataset hash mismatch")
    if manifest.get("version") == 4:
        for name, field in (
            ("feature_provenance.json", "feature_provenance_sha256"),
            ("scoring_provenance.json", "scoring_provenance_sha256"),
        ):
            if file_sha256(directory / name) != manifest[field]:
                raise ValueError("scoring benchmark provenance hash mismatch")
    return cast(dict[str, Any], manifest)


def archive_benchmark(directory: Path) -> dict[str, Any] | None:
    """Copy the current benchmark once, retaining its exact manifest and data bytes."""
    if not (directory / "manifest.json").exists():
        return None
    with advisory_lock(directory / "versions" / ".archive.lock"):
        return _archive_benchmark_locked(directory)


def _archive_benchmark_locked(directory: Path) -> dict[str, Any]:
    if not (directory / "manifest.json").exists():
        raise ValueError("benchmark manifest disappeared during archive")
    manifest = _verified_manifest(directory)
    digest = file_sha256(directory / "manifest.json")
    version = f"dataset-{digest}"
    versions = directory / "versions"
    destination = versions / version
    pointer = versions / "current.json"
    previous = json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else None
    if previous is not None:
        prior = versions / previous["dataset_version"]
        if (
            not prior.is_dir()
            or file_sha256(prior / "manifest.json") != previous["manifest_sha256"]
        ):
            raise ValueError("benchmark history pointer is invalid")
    if destination.exists():
        _verified_manifest(destination)
        if file_sha256(destination / "manifest.json") != digest:
            raise ValueError("immutable benchmark manifest collision")
        status = json.loads((destination / "race_status.json").read_text(encoding="utf-8"))
        if status["dataset_version"] != version or status["manifest_sha256"] != digest:
            raise ValueError("immutable benchmark race status mismatch")
    else:
        versions.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{version}-", dir=versions) as staging_name:
            staging = Path(staging_name)
            for name in ("manifest.json", "coverage.json"):
                shutil.copyfile(directory / name, staging / name)
            for record in manifest["datasets"].values():
                shutil.copyfile(directory / record["path"], staging / record["path"])
            if manifest.get("version") == 4:
                for name in ("feature_provenance.json", "scoring_provenance.json"):
                    shutil.copyfile(directory / name, staging / name)
            _verified_manifest(staging)
            coverage = json.loads((staging / "coverage.json").read_text(encoding="utf-8"))
            previous_rows: dict[tuple[str, str], dict[str, Any]] = {}
            if previous is not None:
                prior_coverage = json.loads((prior / "coverage.json").read_text(encoding="utf-8"))
                previous_rows = {
                    (str(row.get("event_id")), str(row.get("cutoff_kind"))): row
                    for row in prior_coverage["coverage"]
                }
            history = []
            current_keys = set()
            for row in coverage["coverage"]:
                key = (str(row.get("event_id")), str(row.get("cutoff_kind")))
                current_keys.add(key)
                old = previous_rows.get(key)
                if (
                    old is None
                    and previous is not None
                    and row.get("status") == "included"
                    and row.get("tier") == "Gold"
                ):
                    old = {
                        "event_id": row.get("event_id"),
                        "cutoff_kind": row.get("cutoff_kind"),
                        "status": "excluded",
                        "reasons": ["absent_from_previous_benchmark"],
                        "evidence_quality": "not_recorded",
                    }
                transition = (
                    "excluded_to_gold"
                    if old is not None
                    and old.get("status") == "excluded"
                    and row.get("status") == "included"
                    and row.get("tier") == "Gold"
                    else "changed"
                    if old is not None and old != row
                    else "unchanged"
                    if old is not None
                    else "first_observed"
                )
                history.append({"current": row, "previous": old, "transition": transition})
            for key in sorted(previous_rows.keys() - current_keys):
                old = previous_rows[key]
                withdrawn = {
                    "event_id": old.get("event_id"),
                    "cutoff_kind": old.get("cutoff_kind"),
                    "status": "excluded",
                    "reasons": ["absent_from_current_benchmark"],
                }
                history.append(
                    {
                        "current": withdrawn,
                        "previous": old,
                        "transition": "gold_to_excluded"
                        if old.get("status") == "included" and old.get("tier") == "Gold"
                        else "removed",
                    }
                )
            status = {
                "dataset_version": version,
                "manifest_sha256": digest,
                "previous_version": previous["dataset_version"] if previous else None,
                "races": history,
            }
            (staging / "race_status.json").write_bytes(_canonical(status))
            staging.replace(destination)
    current = {"dataset_version": version, "manifest_sha256": digest}
    if previous != current:
        pointer_transition = {
            "previous_version": previous["dataset_version"] if previous else None,
            "dataset_version": version,
            "manifest_sha256": digest,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        with (versions / "history.jsonl").open("ab") as handle:
            handle.write(_canonical(pointer_transition) + b"\n")
        temporary = versions / ".current-pending.json"
        temporary.write_bytes(_canonical(current))
        temporary.replace(pointer)
    return {
        **current,
        "path": destination,
        "gold_race_count": len(manifest["datasets"]["Gold"]["events"]),
        "driver_race_count": manifest["datasets"]["Gold"]["rows"],
    }
