"""Compare scored Gold with its prior version on identical chronological races."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from f1_ml_predictor.models.uncertainty import paired_loss_intervals

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("hist", "catboost")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("previous_report", "previous_predictions", "scored_report", "scored_predictions"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    previous = json.loads(args.previous_report.read_text(encoding="utf-8"))
    scored = json.loads(args.scored_report.read_text(encoding="utf-8"))
    if previous["prediction_sha256"] != digest(args.previous_predictions) or scored[
        "prediction_sha256"
    ] != digest(args.scored_predictions):
        raise ValueError("model predictions differ from their run reports")
    if (
        previous["draws"] != scored["draws"]
        or previous["seed"] != scored["seed"]
        or previous["tier"] != scored["tier"]
    ):
        raise ValueError("benchmark model protocols differ")
    prior_rows = pq.read_table(args.previous_predictions).to_pylist()
    scored_rows = pq.read_table(args.scored_predictions).to_pylist()
    models = {}
    for backend in BACKENDS:
        earlier = previous["comparisons"]["post_qualifying"][backend]
        current = scored["comparisons"]["post_qualifying"][backend]
        if earlier["paired_event_ids"] != current["paired_event_ids"]:
            raise ValueError("benchmark outer folds differ")
        models[backend] = {
            "paired_events": len(current["paired_event_ids"]),
            "previous_metrics": earlier["metrics"],
            "scored_metrics": current["metrics"],
            "scored_minus_previous": paired_loss_intervals(
                [row for row in scored_rows if row["backend"] == backend],
                reference([row for row in prior_rows if row["backend"] == backend]),
                "logistic",
                seed=42,
            ),
        }
    output = {
        "previous_report": args.previous_report.resolve().relative_to(ROOT).as_posix(),
        "scored_report": args.scored_report.resolve().relative_to(ROOT).as_posix(),
        "previous_manifest_sha256": previous["benchmark_manifest_sha256"],
        "scored_manifest_sha256": scored["benchmark_manifest_sha256"],
        "scoring_ledger_sha256": scored["run_metadata"]["scoring_ledger_sha256"],
        "seed": scored["seed"],
        "draws": scored["draws"],
        "protocol": "gold-chronological-v2",
        "unit": "race",
        "models": models,
    }
    target = (
        ROOT
        / "models/experiments/gold/scoring_ablations"
        / scored["benchmark_dataset_sha256"]
        / "version_comparison.json"
    )
    content = json.dumps(output, indent=2, sort_keys=True)
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise ValueError("immutable benchmark version comparison collision")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(content, encoding="utf-8")
    print(target.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
