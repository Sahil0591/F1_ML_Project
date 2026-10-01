"""Run race-paired diagnostic ablations of audited championship features."""

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.scoring import _POINT_FEATURES
from f1_ml_predictor.models.probabilistic import run_probabilistic_files
from f1_ml_predictor.models.uncertainty import paired_loss_intervals
from f1_ml_predictor.trust.evidence import BenchmarkTier

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("hist", "catboost")
MASKS = {
    "without_driver_points": {name for name in _POINT_FEATURES if name.startswith("driver_")},
    "without_constructor_points": {
        name for name in _POINT_FEATURES if name.startswith("constructor_")
    },
    "without_championship_points": set(_POINT_FEATURES),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(rows: list[dict]) -> list[dict]:
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


def masked_benchmark(source: Path, name: str, mask: set[str]) -> Path:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    gold = source / manifest["datasets"]["Gold"]["path"]
    if manifest["version"] != 4 or digest(gold) != manifest["datasets"]["Gold"]["sha256"]:
        raise ValueError("source scoring benchmark is incomplete")
    table = pq.read_table(gold)
    if not all(
        name in table.column_names and f"{name}_missing" in table.column_names for name in mask
    ):
        raise ValueError("source scoring features are incomplete")
    rows = []
    for original in table.to_pylist():
        row = original.copy()
        for feature in mask:
            row[feature] = None
            row[f"{feature}_missing"] = True
        rows.append(row)
    output = ROOT / "data/benchmarks/gold_scoring_ablations_v1" / digest(gold) / name
    output.mkdir(parents=True, exist_ok=True)
    target = output / "gold.parquet"
    if target.exists():
        if pq.read_table(target).to_pylist() != rows:
            raise ValueError("immutable scoring ablation collision")
    else:
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), target, compression="zstd")
    datasets = dict(manifest["datasets"])
    datasets["Gold"] = {**datasets["Gold"], "sha256": digest(target)}
    for tier in ("Silver", "Development"):
        item = datasets[tier]
        replica = output / item["path"]
        if not replica.exists():
            shutil.copyfile(source / item["path"], replica)
        if digest(replica) != item["sha256"]:
            raise ValueError("secondary benchmark differs from source")
    for filename, hash_key in (
        ("coverage.json", "coverage_sha256"),
        ("feature_provenance.json", "feature_provenance_sha256"),
        ("scoring_provenance.json", "scoring_provenance_sha256"),
    ):
        replica = output / filename
        if not replica.exists():
            shutil.copyfile(source / filename, replica)
        if digest(replica) != manifest[hash_key]:
            raise ValueError(f"{filename} differs from source")
    ablated = {
        **manifest,
        "datasets": datasets,
        "diagnostic_ablation": name,
        "diagnostic_source_manifest_sha256": digest(source / "manifest.json"),
        "selection_eligible": False,
    }
    contents = json.dumps(ablated, sort_keys=True, separators=(",", ":")).encode()
    target_manifest = output / "manifest.json"
    if target_manifest.exists() and target_manifest.read_bytes() != contents:
        raise ValueError("immutable scoring ablation manifest collision")
    if not target_manifest.exists():
        target_manifest.write_bytes(contents)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("full_report", type=Path)
    parser.add_argument("full_predictions", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    full = json.loads(args.full_report.read_text(encoding="utf-8"))
    if full["benchmark_manifest_sha256"] != digest(source / "manifest.json"):
        raise ValueError("full model run uses another Gold benchmark")
    if full["tier"] != "Gold" or full["seed"] != 42 or full["draws"] != 4096:
        raise ValueError("full model run must use the frozen scoring diagnostic protocol")
    if full["prediction_sha256"] != digest(args.full_predictions):
        raise ValueError("full model predictions differ from run report")
    full_rows = pq.read_table(args.full_predictions).to_pylist()
    source_gold = json.loads((source / "manifest.json").read_text(encoding="utf-8"))["datasets"][
        "Gold"
    ]["sha256"]
    results = {}
    for name, mask in MASKS.items():
        dataset = masked_benchmark(source, name, mask)
        run_root = ROOT / "models/experiments/gold/scoring_ablations" / source_gold / name
        existing = list(run_root.glob("*/comparison.json"))
        if len(existing) > 1:
            raise ValueError("multiple diagnostic runs require explicit selection")
        if existing:
            report_path = existing[0]
            run_id = report_path.parent.name
        else:
            run_id = uuid.uuid4().hex
            report_path = run_root / run_id / "comparison.json"
        prediction_path = (
            ROOT
            / "data/predictions/scoring_ablations"
            / source_gold
            / name
            / run_id
            / "gold.parquet"
        )
        if existing:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["benchmark_manifest_sha256"] != digest(dataset / "manifest.json") or report[
                "prediction_sha256"
            ] != digest(prediction_path):
                raise ValueError("existing diagnostic run differs from benchmark")
        else:
            report = run_probabilistic_files(
                dataset / "gold.parquet",
                BenchmarkTier.GOLD,
                report_path,
                prediction_path,
                run_id=run_id,
                backends=BACKENDS,
                device="cpu",
                seed=42,
                draws=4096,
            )
        predictions = pq.read_table(prediction_path).to_pylist()
        models = {}
        for backend in BACKENDS:
            comparison = report["comparisons"]["post_qualifying"][backend]
            control = full["comparisons"]["post_qualifying"][backend]
            if comparison["paired_event_ids"] != control["paired_event_ids"]:
                raise ValueError("ablation and full model outer folds differ")
            rows = [row for row in predictions if row["backend"] == backend]
            full_backend = [row for row in full_rows if row["backend"] == backend]
            models[backend] = {
                "paired_events": len(comparison["paired_event_ids"]),
                "metrics": comparison["metrics"],
                "versus_full": paired_loss_intervals(
                    rows, reference(full_backend), "logistic", seed=42
                ),
            }
        results[name] = {
            "report": report_path.relative_to(ROOT).as_posix(),
            "prediction": prediction_path.relative_to(ROOT).as_posix(),
            "models": models,
        }
        print(name, report_path.relative_to(ROOT).as_posix(), flush=True)
    summary = {
        "source_gold_sha256": source_gold,
        "source_manifest_sha256": digest(source / "manifest.json"),
        "scoring_ledger_sha256": full["run_metadata"]["scoring_ledger_sha256"],
        "protocol": "gold-chronological-v2",
        "seed": 42,
        "draws": 4096,
        "backends": list(BACKENDS),
        "diagnostic_only": True,
        "full_report": args.full_report.resolve().relative_to(ROOT).as_posix(),
        "stages": results,
    }
    output = ROOT / "models/experiments/gold/scoring_ablations" / source_gold / "summary.json"
    contents = json.dumps(summary, sort_keys=True, indent=2)
    if output.exists() and output.read_text(encoding="utf-8") != contents:
        raise ValueError("immutable scoring ablation summary collision")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_text(contents, encoding="utf-8")
    print("summary", output.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
