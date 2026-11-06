"""Freeze archived ECMWF race-time forecasts for the current Gold coverage report.

Each race uses the run released (with the documented margin) before its
post-qualifying cutoff; races before the archive start stay missing. Race
starts come from the report's audited direct catalog, not the Jolpica schedule.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from f1_ml_predictor.benchmarks.builder import _safe_file
from f1_ml_predictor.prediction.workspace import load_audited_history
from f1_ml_predictor.trust.weather_history import audit_historical_weather


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_report", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    report = json.loads(args.coverage_report.read_text(encoding="utf-8"))
    direct = json.loads(
        _safe_file(root, report["direct_catalog"], report["direct_catalog_sha256"]).read_text(
            encoding="utf-8"
        )
    )
    starts = {
        f"season={item['season']}/round={item['round']:02d}": datetime.fromisoformat(
            item["race_start"]
        )
        for item in direct["candidates"]
    }
    history, _, _ = load_audited_history(root)
    events = {
        event_id: (rows[0]["prediction_timestamp"], starts[event_id])
        for event_id, rows in history.by_event.items()
    }
    print(json.dumps(audit_historical_weather(root, events), indent=2))


if __name__ == "__main__":
    main()
