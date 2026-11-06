"""Score official and shadow race predictions against audited Gold outcomes.

Each shadow file is scored once its race is in the Gold registry. Per variant:
winner log loss, whether the favourite won, podium Brier, podium hits among the
three highest podium probabilities and finishing MAE on classified drivers.
Each shadow is also averaged against the official run over the races where both
were published (negative is better for every metric except favourite_won and
podium_hits).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import _safe_file

VARIANTS = ("official", "two_stage", "weather", "wet_skill", "strategy")


def _outcomes(root: Path) -> dict[str, Path]:
    registry = json.loads((root / "data/benchmarks/gold_core_registry.json").read_text("utf-8"))
    return {
        item["event_id"]: _safe_file(root, item["outcomes"]["path"], item["outcomes"]["sha256"])
        for item in registry["races"]
    }


def _gold_id(driver: str, gold: set[str]) -> str:
    """Capture IDs such as arvid_lindblad map to a unique Gold suffix such as lindblad."""
    if driver in gold:
        return driver
    matches = [item for item in gold if driver.endswith(f"_{item}")]
    if len(matches) != 1:
        raise ValueError(f"cannot match predicted driver {driver} to a Gold driver")
    return matches[0]


def _score(drivers: dict[str, Any], labels: dict[str, dict[str, Any]]) -> dict[str, float]:
    gold = set(labels)
    values = {_gold_id(driver, gold): item for driver, item in drivers.items()}
    winner = next(driver for driver, row in labels.items() if row["winner"])
    total = sum(item["winner"] for item in values.values())
    favourite = max(values, key=lambda driver: values[driver]["winner"])
    podium = sorted(values, key=lambda driver: -values[driver]["podium"])[:3]
    classified = [driver for driver, row in labels.items() if row["position"] is not None]
    return {
        "winner_log_loss": -math.log(max(values[winner]["winner"] / total, 1e-15)),
        "favourite_won": float(favourite == winner),
        "podium_brier": sum(
            (values[driver]["podium"] - float(bool(labels[driver]["podium"]))) ** 2
            for driver in values
        )
        / len(values),
        "podium_hits": float(sum(bool(labels[driver]["podium"]) for driver in podium)),
        "position_mae": sum(
            abs(values[driver]["expected"] - labels[driver]["position"]) for driver in classified
        )
        / len(classified),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    outcomes = _outcomes(root)
    races = []
    for path in sorted((root / "data/predictions/development/shadows").glob("*/*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        event = payload["event_id"]
        if event not in outcomes:
            races.append({"event_id": event, "status": "awaiting_gold_outcome"})
            continue
        labels = {row["driver_id"]: row for row in pq.read_table(outcomes[event]).to_pylist()}
        scores = {
            name: _score(payload["variants"][name]["drivers"], labels)
            for name in VARIANTS
            if "drivers" in payload["variants"].get(name, {})
        }
        races.append(
            {
                "event_id": event,
                "cutoff_kind": payload["cutoff_kind"],
                "official_run_id": payload["official_run_id"],
                "forecast_rain_mm": (payload["forecast"].get("values") or {}).get(
                    "forecast_race_precipitation_mm"
                ),
                "winner": next(driver for driver, row in labels.items() if row["winner"]),
                "scores": scores,
            }
        )
    totals = {}
    for name in VARIANTS[1:]:
        paired = [race for race in races if {"official", name} <= set(race.get("scores", {}))]
        if not paired:
            continue
        totals[name] = {
            "races": len(paired),
            **{
                f"{metric}_minus_official": sum(
                    race["scores"][name][metric] - race["scores"]["official"][metric]
                    for race in paired
                )
                / len(paired)
                for metric in paired[0]["scores"]["official"]
            },
        }
    print(json.dumps({"races": races, "mean_paired_with_official": totals}, indent=2))


if __name__ == "__main__":
    main()
