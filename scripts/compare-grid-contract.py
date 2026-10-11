"""Pair the provisional-grid addendum against the frozen pre_race contract.

Both contracts must already be evaluated (`evaluate-cutoffs --contracts
provisional_grid pre_race`). The script reloads their cached out-of-fold records,
recomputes the selected pipeline's per-race marginals, and reports race-paired
loss deltas (provisional_grid minus pre_race) plus the drivers who lost five or
more grid places against qualifying.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from f1_ml_predictor.prediction.contracts import build_contract_datasets
from f1_ml_predictor.prediction.evaluation import BOOTSTRAP_RESAMPLES, _race_losses, analyse
from f1_ml_predictor.prediction.protocol import GRID_CONTRACT
from f1_ml_predictor.prediction.workspace import load_audited_history, out_of_fold_path

BASE = "pre_race"
PENALTY_PLACES = 5


def _selected(root: Path, history: Any, contract: str, sha: str, seed: int) -> dict[str, Any]:
    path = out_of_fold_path(root, history, contract, sha, seed=seed)
    if not path.exists():
        raise SystemExit(f"{contract} is not evaluated yet; run evaluate-cutoffs first")
    records = joblib.load(path)
    outputs: dict[str, list[dict[str, Any]]] = {}
    report = analyse(records, seed=seed, outputs=outputs)
    return {
        "records": {record.event_id: record for record in records},
        "values": dict(
            zip((record.event_id for record in records), outputs["selected_pipeline"], strict=True)
        ),
        "metrics": report["models"]["selected_pipeline"]["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    history, dnf_version, sources = load_audited_history(root)
    datasets = build_contract_datasets(root, history, dnf_version, sources)
    if GRID_CONTRACT not in datasets:
        raise SystemExit("no provisional-grid audit for the current Gold version")
    runs = {
        contract: _selected(
            root, history, contract, datasets[contract]["manifest"]["dataset_sha256"], args.seed
        )
        for contract in (GRID_CONTRACT, BASE)
    }
    events = sorted(set(runs[GRID_CONTRACT]["records"]) & set(runs[BASE]["records"]))
    rng = np.random.default_rng(args.seed)
    paired = {}
    for metric in ("winner_log_loss", "podium_brier", "position_mae"):
        deltas = np.asarray(
            [
                _race_losses(
                    runs[GRID_CONTRACT]["records"][event], runs[GRID_CONTRACT]["values"][event]
                )[metric]
                - _race_losses(runs[BASE]["records"][event], runs[BASE]["values"][event])[metric]
                for event in events
            ]
        )
        sampled = rng.choice(deltas, size=(BOOTSTRAP_RESAMPLES, len(deltas)), replace=True)
        lower, upper = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
        paired[metric] = {
            "mean_loss_delta": float(deltas.mean()),
            "bootstrap_95_percent_interval": [float(lower), float(upper)],
            "paired_races": len(deltas),
        }
    rows = {(row["event_id"], row["driver_id"]): row for row in datasets[GRID_CONTRACT]["rows"]}
    penalised = []
    for event in events:
        record = runs[GRID_CONTRACT]["records"][event]
        for index, driver in enumerate(record.drivers):
            row = rows[(event, driver)]
            lost = row["grid_places_lost"]
            if lost is None or lost < PENALTY_PLACES or np.isnan(record.positions[index]):
                continue
            base_index = runs[BASE]["records"][event].drivers.index(driver)
            penalised.append(
                {
                    "event_id": event,
                    "driver_id": driver,
                    "qualifying_position": row["qualifying_position"],
                    "grid_position": row["grid_position"],
                    "pit_lane_start": row["pit_lane_start"] == 1.0,
                    "finish_position": float(record.positions[index]),
                    "expected_provisional_grid": float(
                        runs[GRID_CONTRACT]["values"][event]["expected"][index]
                    ),
                    "expected_pre_race": float(runs[BASE]["values"][event]["expected"][base_index]),
                }
            )
    errors = {
        contract: float(
            np.mean(
                [abs(item[f"expected_{contract}"] - item["finish_position"]) for item in penalised]
            )
        )
        for contract in (GRID_CONTRACT, BASE)
    }
    summary = {
        "contracts": [GRID_CONTRACT, BASE],
        "paired_races": len(events),
        "paired_selected_pipeline_delta": paired,
        "selected_pipeline_metrics": {
            contract: {
                "winner_log_loss": run["metrics"]["winner"]["log_loss"],
                "podium_brier": run["metrics"]["podium"]["brier_score"],
                "position_mae": run["metrics"]["finishing_position"]["mean_absolute_error"],
            }
            for contract, run in runs.items()
        },
        "penalised_drivers": {
            "threshold_places": PENALTY_PLACES,
            "classified_drivers": len(penalised),
            "mean_absolute_error": errors,
            "mean_expected_minus_qualifying": {
                contract: float(
                    np.mean(
                        [
                            item[f"expected_{contract}"] - item["qualifying_position"]
                            for item in penalised
                        ]
                    )
                )
                for contract in (GRID_CONTRACT, BASE)
            },
            "mean_finish_minus_qualifying": float(
                np.mean(
                    [item["finish_position"] - item["qualifying_position"] for item in penalised]
                )
            ),
            "drivers": penalised,
        },
    }
    output = (
        root
        / "models/experiments/gold"
        / history.version.dataset_version
        / "cutoff_v3"
        / GRID_CONTRACT
        / f"comparison-{history.grid_audit}.json"
    )
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "penalised_drivers"}, indent=2
        )
    )
    print(
        json.dumps(
            {key: value for key, value in summary["penalised_drivers"].items() if key != "drivers"},
            indent=2,
        )
    )
    print(output.relative_to(root).as_posix())


if __name__ == "__main__":
    main()
