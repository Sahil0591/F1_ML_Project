"""Score official and shadow race predictions against audited Gold outcomes.

Each shadow file is scored once its race is in the Gold registry. Per variant:
winner log loss, whether the favourite won, podium Brier, podium hits among the
three highest podium probabilities and finishing MAE on classified drivers.
Each race also reports the conditions the shadows were given (the forecast)
against what happened: tyres fitted and the circuit weather station.
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
from f1_ml_predictor.prediction.shadow import wet_forecast
from f1_ml_predictor.prediction.workspace import load_audited_history

VARIANTS = ("official", "two_stage", "weather", "wet_skill", "strategy", "provisional_grid")


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
    history, _, _ = load_audited_history(root)
    strategy = history.strategy
    races = []
    for path in sorted((root / "data/predictions/development/shadows").glob("*/*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        event = payload["event_id"]
        if payload.get("rehearsal") is not None:
            # Rebuilt after the race, so it is not a test on an unseen race.
            continue
        if event not in outcomes:
            races.append({"event_id": event, "status": "awaiting_gold_outcome"})
            continue
        labels = {row["driver_id"]: row for row in pq.read_table(outcomes[event]).to_pylist()}
        scores = {
            name: _score(payload["variants"][name]["drivers"], labels)
            for name in VARIANTS
            if "drivers" in payload["variants"].get(name, {})
        }
        forecast = payload["forecast"]
        values = forecast.get("values") or {}
        actual = strategy.get(event, {}).get("race")
        races.append(
            {
                "event_id": event,
                "cutoff_kind": payload["cutoff_kind"],
                "official_run_id": payload["official_run_id"],
                "winner": next(driver for driver, row in labels.items() if row["winner"]),
                # What the shadows were told, against what happened at the track.
                "conditions": {
                    "forecast_wet": wet_forecast(forecast),
                    "forecast_rain_before_mm": values.get("forecast_pre_race_precipitation_mm"),
                    "forecast_rain_during_mm": values.get("forecast_race_precipitation_mm"),
                    "forecast_temperature_c": values.get("forecast_temperature_2m"),
                    "tyres_wet": None if actual is None else actual["wet"],
                    "station_rain_fraction": None
                    if actual is None
                    else actual.get("track_rain_fraction"),
                    "station_temperature_c": None
                    if actual is None
                    else actual.get("track_air_temperature"),
                    "status": "checked"
                    if actual is not None
                    else "run audit-strategy-history.py to check against the track",
                },
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
