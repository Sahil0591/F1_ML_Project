"""Verified access to immutable Gold versions and their frozen comparison runs.

Nothing here selects a validated model. Reference runs only rank development
candidates when the frozen task gates report ``no_selection``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import _safe_file, file_sha256
from f1_ml_predictor.benchmarks.rolling import _event
from f1_ml_predictor.benchmarks.scoring import SCORING_VERSION, verify_scoring_manifest
from f1_ml_predictor.benchmarks.versioning import _verified_manifest
from f1_ml_predictor.models.backtest import _feature_columns
from f1_ml_predictor.models.boosting import BACKENDS
from f1_ml_predictor.models.protocol import PROTOCOL

_SCORING_ROOT = Path("data/benchmarks") / SCORING_VERSION.replace("-", "_")
_DNF_ROOT = Path("data/benchmarks/gold_core_binary_dnf_v1")
_EXPERIMENTS = Path("models/experiments/gold")
POSITION_TASKS = (
    ("winner", "log_loss"),
    ("podium", "brier_score"),
    ("finishing_position", "mean_absolute_error"),
)


@dataclass(frozen=True)
class GoldVersion:
    """One verified immutable benchmark version and its Gold rows."""

    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    dataset_version: str
    schema: pa.Schema
    rows: list[dict[str, Any]]
    feature_columns: tuple[str, ...]

    @property
    def events(self) -> dict[tuple[int, int], list[dict[str, Any]]]:
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            grouped[_event(row["event_id"])].append(row)
        return dict(grouped)


def _latest_pointer(pattern_root: Path, pattern: str) -> Path:
    """Choose the version pointer whose history records the latest publication."""
    candidates = []
    for pointer in pattern_root.glob(pattern):
        history = pointer.with_name("history.jsonl")
        lines = history.read_text(encoding="utf-8").splitlines() if history.exists() else []
        recorded = (
            datetime.fromisoformat(json.loads(lines[-1])["recorded_at"])
            if lines
            else datetime.min.replace(tzinfo=UTC)
        )
        candidates.append((recorded, pointer.as_posix(), pointer))
    if not candidates:
        raise ValueError(f"no versioned benchmark found under {pattern_root.as_posix()}")
    return max(candidates)[2]


def latest_gold_directory(root: Path) -> Path:
    """Return the newest immutable scoring Gold version for the current feature schema."""
    pointer = _latest_pointer(root / _SCORING_ROOT, "*/*/versions/current.json")
    current = json.loads(pointer.read_text(encoding="utf-8"))
    return pointer.parent / str(current["dataset_version"])


def latest_dnf_directory(root: Path) -> Path:
    """Return the newest immutable audited binary DNF version."""
    pointer = _latest_pointer(root / _DNF_ROOT, "*/benchmark/versions/current.json")
    current = json.loads(pointer.read_text(encoding="utf-8"))
    return pointer.parent / str(current["dataset_version"])


def load_gold_version(directory: Path) -> GoldVersion:
    """Verify every hash of an immutable version before any row is used."""
    directory = directory.resolve()
    manifest = _verified_manifest(directory)
    verify_scoring_manifest(directory, manifest)
    digest = file_sha256(directory / "manifest.json")
    if directory.name.startswith("dataset-") and directory.name != f"dataset-{digest}":
        raise ValueError("immutable dataset directory does not match its manifest hash")
    gold = manifest["datasets"]["Gold"]
    table = pq.read_table(directory / gold["path"])
    rows = table.to_pylist()
    if len(rows) != gold["rows"] or {row["benchmark_tier"] for row in rows} != {"Gold"}:
        raise ValueError("Gold version row count or tier differs from its manifest")
    if {row["event_id"] for row in rows} != set(gold["events"]):
        raise ValueError("Gold version event set differs from its manifest")
    columns = _feature_columns(rows)
    if list(columns) != manifest["feature_columns"]:
        raise ValueError("Gold version predictor schema differs from its manifest")
    return GoldVersion(
        directory, manifest, digest, f"dataset-{digest}", table.schema, rows, columns
    )


def audited_outcomes(root: Path, version: GoldVersion) -> dict[tuple[int, int], str]:
    """Bind each Gold event to its hash-verified audited outcome file."""
    registry_path = root / "data/benchmarks/gold_core_registry.json"
    if file_sha256(registry_path) != version.manifest["catalog_sha256"]:
        raise ValueError("audited Gold registry changed since the benchmark version")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    outcomes: dict[tuple[int, int], str] = {}
    for item in registry["races"]:
        specification = item["outcomes"]
        _safe_file(root, specification["path"], specification["sha256"])
        outcomes[_event(item["event_id"])] = specification["sha256"]
    if set(outcomes) != set(version.events):
        raise ValueError("some Gold events lack hash-verified audited outcomes")
    return outcomes


def reference_run(root: Path, version: GoldVersion) -> tuple[Path, dict[str, Any]]:
    """Find the latest frozen-protocol comparison for exactly this dataset version."""
    candidates = []
    for path in (root / _EXPERIMENTS / version.dataset_version / "runs").glob("*/comparison.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        metadata = report.get("run_metadata", {})
        parameters = metadata.get("model_parameters", {})
        if (
            report.get("diagnostic_only")
            or report.get("benchmark_manifest_sha256") != version.manifest_sha256
            or report.get("evaluation_protocol", {}).get("version") != PROTOCOL["version"]
            or sorted(parameters.get("backends", [])) != sorted(BACKENDS)
            or parameters.get("calibration_events") != 1
            or metadata.get("calibration_method") != "sigmoid"
        ):
            continue
        candidates.append((metadata["created_at"], path.as_posix(), path, report))
    if not candidates:
        raise ValueError(f"no frozen comparison run exists for {version.dataset_version}")
    _, _, path, report = max(candidates, key=lambda item: (item[0], item[1]))
    return path, report


def choose_position_backend(report: dict[str, Any], cutoff_kind: str) -> dict[str, Any]:
    """Rank joint candidates when no task is selected; a selection always wins.

    The joint sampler needs one position model for coherent winner, podium and
    finish marginals, so the development rule is the lowest mean rank across
    those three primary losses, tied by winner log loss.
    """
    comparisons = report["comparisons"][cutoff_kind]
    selections = report["task_selection"][cutoff_kind]
    selected = {
        selections[task]["selected_backend"]
        for task, _ in POSITION_TASKS
        if selections[task]["status"] == "provisional"
    }
    metrics = {
        backend: {task: comparison["metrics"][task].get(metric) for task, metric in POSITION_TASKS}
        for backend, comparison in comparisons.items()
    }
    baselines = next(iter(comparisons.values()))["baselines"]
    baseline_metrics = {
        name: {task: values[task].get(metric) for task, metric in POSITION_TASKS}
        for name, values in baselines.items()
    }
    if len(selected) == 1:
        backend = next(iter(selected))
        return {
            "backend": backend,
            "status": "provisional_selection",
            "rule": "frozen task selection",
            "candidate_metrics": metrics,
            "baseline_metrics": baseline_metrics,
        }
    ranks: dict[str, list[int]] = defaultdict(list)
    for task, _ in POSITION_TASKS:
        ordered = sorted(
            (values[task], backend)
            for backend, values in metrics.items()
            if values[task] is not None
        )
        for rank, (_, backend) in enumerate(ordered, 1):
            ranks[backend].append(rank)
    eligible = [backend for backend, values in ranks.items() if len(values) == len(POSITION_TASKS)]
    if not eligible:
        raise ValueError("no joint candidate has all position task metrics")
    backend = min(
        eligible,
        key=lambda name: (sum(ranks[name]) / len(ranks[name]), metrics[name]["winner"], name),
    )
    return {
        "backend": backend,
        "status": "development_candidate",
        "rule": (
            "no task passed formal selection; lowest mean rank across winner log loss, "
            "podium Brier score and finish MAE among joint candidates"
        ),
        "mean_rank": {name: sum(values) / len(values) for name, values in ranks.items()},
        "candidate_metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "task_selection": {
            task: selections[task]["status"] for task in (*dict(POSITION_TASKS), "dnf")
        },
    }


def choose_dnf_model(report: dict[str, Any], cutoff_kind: str) -> dict[str, Any]:
    """Use a selected DNF backend, otherwise the lowest observed audited DNF Brier."""
    selection = report["task_selection"][cutoff_kind]["dnf"]
    if selection["status"] == "provisional":
        return {"model": selection["selected_backend"], "status": "provisional_selection"}
    observed = selection.get("observed_lowest_loss")
    if observed is None:
        raise ValueError("the audited DNF comparison has no evaluated DNF candidate")
    if observed["model"] == "heuristic":
        raise ValueError("heuristic DNF is not a supported development model")
    return {
        "model": observed["model"],
        "status": "development_candidate",
        "rule": "no DNF selection; lowest observed audited DNF Brier score",
        "observed_brier_score": observed["value"],
    }
