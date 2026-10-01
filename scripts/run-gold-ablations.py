"""Compare cumulative Gold feature groups on identical chronological races."""

import argparse
import json
import uuid
from pathlib import Path

import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.versioning import archive_benchmark
from f1_ml_predictor.models.probabilistic import run_probabilistic_files
from f1_ml_predictor.models.uncertainty import paired_loss_intervals
from f1_ml_predictor.trust.evidence import BenchmarkTier

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "qualifying_only",
    "qualifying_plus_recent_form",
    "plus_constructor_form",
    "plus_teammate_features",
    "plus_championship_context",
    "plus_practice_where_available",
    "plus_weather_where_available",
)
BACKENDS = ("hist", "catboost")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    source_gold = source_manifest["datasets"]["Gold"]["sha256"]
    ablations = ROOT / "data/benchmarks/gold_diagnostic_ablations_v1" / source_gold
    summary = {"source_gold_sha256": source_gold, "stages": {}, "backends": list(BACKENDS)}
    previous: dict[str, list[dict]] = {}
    expected_events: list[str] | None = None
    for stage in (*STAGES, "full"):
        directory = source if stage == "full" else ablations / stage
        snapshot = archive_benchmark(directory)
        if snapshot is None:
            raise ValueError(f"missing ablation benchmark: {directory}")
        run_id = uuid.uuid4().hex
        report_path = (
            ROOT
            / "models/experiments/gold/ablations"
            / source_gold
            / stage
            / run_id
            / "comparison.json"
        )
        prediction_path = (
            ROOT / "data/predictions/ablations" / source_gold / stage / run_id / "gold.parquet"
        )
        report = run_probabilistic_files(
            snapshot["path"] / "gold.parquet",
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
        stage_result = {
            "run_id": run_id,
            "dataset_manifest_hash": snapshot["manifest_sha256"],
            "report_path": report_path.relative_to(ROOT).as_posix(),
            "prediction_sha256": report["prediction_sha256"],
            "models": {},
        }
        for backend in BACKENDS:
            comparison = report["comparisons"]["post_qualifying"][backend]
            events = comparison["paired_event_ids"]
            if expected_events is None:
                expected_events = events
            elif events != expected_events:
                raise ValueError("ablation stages have different outer race folds")
            rows = [row for row in predictions if row["backend"] == backend]
            result = {
                "paired_events": len(events),
                "metrics": comparison["metrics"],
                "versus_previous_group": None,
            }
            if backend in previous:
                result["versus_previous_group"] = paired_loss_intervals(
                    rows, _as_reference(previous[backend]), "logistic", seed=42
                )
            stage_result["models"][backend] = result
            previous[backend] = rows
        summary["stages"][stage] = stage_result
        print(stage, report_path.relative_to(ROOT).as_posix(), flush=True)
    output = ROOT / "models/experiments/gold/ablations" / source_gold / "summary.json"
    if output.exists():
        raise ValueError("ablation summary already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    print("summary", output.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
