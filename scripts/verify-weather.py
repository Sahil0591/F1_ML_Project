"""Check archived forecasts against what happened at each Gold race.

Three sources are compared for every race with both a weather and a strategy
audit: the archived forecast (what the model sees), tyres actually fitted
(intermediates or wets: the wet label), and the circuit weather station (share
of readings with rain falling, and air temperature). Disagreements between the
tyre label and the station are listed for review instead of being resolved
silently. The report is printed and frozen under its content hash.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from f1_ml_predictor.prediction.shadow import wet_forecast
from f1_ml_predictor.prediction.workspace import load_audited_history
from f1_ml_predictor.trust.historical import _immutable, _json

# The station says it rained in at least this share of readings.
STATION_WET = 0.25
# Fitted wets with rain in under this share of readings needs a look.
STATION_DRY = 0.05
TEMPERATURE_REVIEW_C = 5.0


def _table(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, int]:
    counts = {"both_wet": 0, "only_" + left: 0, "only_" + right: 0, "both_dry": 0}
    for row in rows:
        a, b = row[left], row[right]
        if a is None or b is None:
            continue
        key = (
            "both_wet" if a and b else f"only_{left}" if a else f"only_{right}" if b else "both_dry"
        )
        counts[key] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    history, _, _ = load_audited_history(root)
    if history.weather_audit is None or history.strategy_audit is None:
        raise SystemExit("run audit-historical-weather.py and audit-strategy-history.py first")
    rows = []
    for event_id in sorted(history.strategy):
        forecast = history.weather.get(event_id)
        if forecast is None or forecast.get("status") != "audited":
            continue
        race = history.strategy[event_id]["race"]
        values = forecast["values"]
        fraction = race.get("track_rain_fraction")
        rows.append(
            {
                "event_id": event_id,
                "forecast_wet": wet_forecast(forecast),
                "forecast_rain_before_mm": values["forecast_pre_race_precipitation_mm"],
                "forecast_rain_during_mm": values["forecast_race_precipitation_mm"],
                "tyres_wet": race["wet"],
                "station_rain_fraction": fraction,
                "station_wet": None if fraction is None else fraction >= STATION_WET,
                "forecast_temperature": values["forecast_temperature_2m"],
                "station_temperature": race.get("track_air_temperature"),
            }
        )
    review = [
        row
        for row in rows
        if row["station_rain_fraction"] is not None
        and (
            (row["tyres_wet"] and row["station_rain_fraction"] < STATION_DRY)
            or (not row["tyres_wet"] and row["station_rain_fraction"] >= STATION_WET)
        )
    ]
    errors = [
        row["forecast_temperature"] - row["station_temperature"]
        for row in rows
        if row["station_temperature"] is not None
    ]
    report = {
        "weather_audit": history.weather_audit,
        "strategy_audit": history.strategy_audit,
        "races": len(rows),
        "forecast_vs_tyres": _table(rows, "forecast_wet", "tyres_wet"),
        "forecast_vs_station": _table(rows, "forecast_wet", "station_wet"),
        "tyres_vs_station": _table(rows, "tyres_wet", "station_wet"),
        "temperature": {
            "races": len(errors),
            "mean_error_c": sum(errors) / len(errors) if errors else None,
            "mean_absolute_error_c": sum(abs(e) for e in errors) / len(errors) if errors else None,
            "off_by_more_than_5c": [
                {
                    "event_id": row["event_id"],
                    "forecast": row["forecast_temperature"],
                    "station": row["station_temperature"],
                }
                for row in rows
                if row["station_temperature"] is not None
                and abs(row["forecast_temperature"] - row["station_temperature"])
                > TEMPERATURE_REVIEW_C
            ],
        },
        "tyres_and_station_disagree": review,
        "rows": rows,
    }
    content = _json(report)
    path = (
        root
        / "data/benchmarks/historical_audit"
        / f"weather-verification-{hashlib.sha256(content).hexdigest()}.json"
    )
    _immutable(path, content)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"report: {path.relative_to(root).as_posix()}")


if __name__ == "__main__":
    main()
