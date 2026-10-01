"""Run fixed-protocol, race-paired Gold feature group diagnostics."""

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.enrichment import (
    _FORM_FEATURES,
    _POINT_FEATURES,
    _PRACTICE_FEATURES,
)
from f1_ml_predictor.models.probabilistic import run_probabilistic_files
from f1_ml_predictor.models.uncertainty import paired_loss_intervals
from f1_ml_predictor.trust.evidence import BenchmarkTier

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("hist", "catboost")
STAGES = {
    "rolling_baseline": set(),
    "constructor_only": set(_FORM_FEATURES),
    "practice_only": set(_PRACTICE_FEATURES),
    "grid_only": {"grid_position"},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_reference(rows: list[dict]) -> list[dict]:
    return [
        {
            **row,
            "logistic_winner_probability": row["winner_probability"],
            "logistic_podium_probability": row["podium_probability"],
            "logistic_dnf_probability": row["dnf_probability"],
            "linear_finish_position": row["expected_position"],
        }
        for row in rows
    ]


def _masked(source: Path, name: str, active: set[str]) -> Path:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    source_gold = source / manifest["datasets"]["Gold"]["path"]
    if digest(source_gold) != manifest["datasets"]["Gold"]["sha256"]:
        raise ValueError("enriched Gold bytes differ from their manifest")
    table = pq.read_table(source_gold)
    target_dir = ROOT / "data/benchmarks/gold_enrichment_ablations_v1" / digest(source_gold) / name
    target_dir.mkdir(parents=True, exist_ok=True)
    masked = []
    to_mask = (
        set(_FORM_FEATURES) | set(_POINT_FEATURES) | set(_PRACTICE_FEATURES) | {"grid_position"}
    ) - active
    for original in table.to_pylist():
        row = original.copy()
        for feature in to_mask:
            row[feature] = None
            row[f"{feature}_missing"] = True
        masked.append(row)
    target = target_dir / "gold.parquet"
    if target.exists():
        if pq.read_table(target).to_pylist() != masked:
            raise ValueError("immutable masked benchmark collision")
    else:
        pq.write_table(
            pa.Table.from_pylist(masked, schema=table.schema), target, compression="zstd"
        )
    datasets = dict(manifest["datasets"])
    datasets["Gold"] = {**datasets["Gold"], "sha256": digest(target)}
    for tier in ("Silver", "Development"):
        original = source / datasets[tier]["path"]
        replica = target_dir / datasets[tier]["path"]
        if not replica.exists():
            shutil.copyfile(original, replica)
        if digest(replica) != datasets[tier]["sha256"]:
            raise ValueError("secondary benchmark differs from the source")
    coverage = target_dir / "coverage.json"
    if not coverage.exists():
        shutil.copyfile(source / "coverage.json", coverage)
    if digest(coverage) != manifest["coverage_sha256"]:
        raise ValueError("masked benchmark coverage differs from the source")
    ablated = {
        **manifest,
        "datasets": datasets,
        "diagnostic_ablation": name,
        "diagnostic_source_manifest_sha256": digest(source / "manifest.json"),
        "selection_eligible": False,
    }
    manifest_bytes = json.dumps(ablated, sort_keys=True, separators=(",", ":")).encode()
    destination = target_dir / "manifest.json"
    if destination.exists() and destination.read_bytes() != manifest_bytes:
        raise ValueError("immutable ablation manifest collision")
    if not destination.exists():
        destination.write_bytes(manifest_bytes)
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("full_report", type=Path)
    parser.add_argument("full_predictions", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    full = json.loads(args.full_report.read_text(encoding="utf-8"))
    if full["benchmark_manifest_sha256"] != digest(source / "manifest.json"):
        raise ValueError("full model run uses another Gold benchmark")
    if digest(args.full_predictions) != full["prediction_sha256"]:
        raise ValueError("full model predictions differ from their run report")
    full_predictions = pq.read_table(args.full_predictions).to_pylist()
    source_gold = source_manifest["datasets"]["Gold"]["sha256"]
    results: dict[str, dict] = {}
    baseline: dict[str, list[dict]] = {}
    for name, active in STAGES.items():
        dataset = _masked(source, name, active)
        run_root = ROOT / "models/experiments/gold/enrichment_ablations" / source_gold / name
        existing = list(run_root.glob("*/comparison.json"))
        if len(existing) > 1:
            raise ValueError("multiple diagnostic runs require explicit selection")
        if existing:
            report_path = existing[0]
            run_id = report_path.parent.name
            prediction_path = (
                ROOT
                / "data/predictions/enrichment_ablations"
                / source_gold
                / name
                / run_id
                / "gold.parquet"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["benchmark_manifest_sha256"] != digest(dataset / "manifest.json") or report[
                "prediction_sha256"
            ] != digest(prediction_path):
                raise ValueError("existing diagnostic run differs from the masked benchmark")
        else:
            run_id = uuid.uuid4().hex
            report_path = run_root / run_id / "comparison.json"
            prediction_path = (
                ROOT
                / "data/predictions/enrichment_ablations"
                / source_gold
                / name
                / run_id
                / "gold.parquet"
            )
            report = run_probabilistic_files(
                dataset / "gold.parquet",
                BenchmarkTier.GOLD,
                report_path,
                prediction_path,
                run_id=run_id,
                backends=BACKENDS,
                device="cpu",
                seed=42,
                draws=512,
            )
        predictions = pq.read_table(prediction_path).to_pylist()
        result = {
            "report": report_path.relative_to(ROOT).as_posix(),
            "prediction": prediction_path.relative_to(ROOT).as_posix(),
            "models": {},
        }
        for backend in BACKENDS:
            comparison = report["comparisons"]["post_qualifying"][backend]
            full_comparison = full["comparisons"]["post_qualifying"][backend]
            if comparison["paired_event_ids"] != full_comparison["paired_event_ids"]:
                raise ValueError("ablation and full model outer folds differ")
            rows = [row for row in predictions if row["backend"] == backend]
            if name == "rolling_baseline":
                baseline[backend] = rows
            result["models"][backend] = {
                "paired_events": len(comparison["paired_event_ids"]),
                "metrics": comparison["metrics"],
                "versus_full": paired_loss_intervals(
                    rows,
                    _as_reference([row for row in full_predictions if row["backend"] == backend]),
                    "logistic",
                    seed=42,
                ),
            }
        results[name] = result
        print(name, result["report"], flush=True)
    for backend in BACKENDS:
        full_rows = [row for row in full_predictions if row["backend"] == backend]
        results.setdefault(
            "full",
            {"report": args.full_report.resolve().relative_to(ROOT).as_posix(), "models": {}},
        )["models"][backend] = {
            "paired_events": len(
                full["comparisons"]["post_qualifying"][backend]["paired_event_ids"]
            ),
            "metrics": full["comparisons"]["post_qualifying"][backend]["metrics"],
            "versus_rolling_baseline": paired_loss_intervals(
                full_rows, _as_reference(baseline[backend]), "logistic", seed=42
            ),
        }
    summary = {
        "source_gold_sha256": source_gold,
        "protocol": "gold-chronological-v2",
        "seed": 42,
        "draws": 512,
        "backends": list(BACKENDS),
        "diagnostic_only": True,
        "stages": results,
    }
    output = ROOT / "models/experiments/gold/enrichment_ablations" / source_gold / "summary.json"
    content = json.dumps(summary, sort_keys=True, indent=2)
    if output.exists() and output.read_text(encoding="utf-8") != content:
        raise ValueError("immutable ablation summary collision")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_text(content, encoding="utf-8")
    print("summary", output.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
