"""Freeze OpenF1 tyre, pit and practice evidence for the current Gold coverage report.

Race starts come from the report's audited direct catalog. Requests are retained
and reused, so an interrupted run resumes where it stopped.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from f1_ml_predictor.benchmarks.builder import _safe_file
from f1_ml_predictor.prediction.workspace import load_audited_history
from f1_ml_predictor.trust.strategy_history import audit_strategy_history


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
        event_id: (starts[event_id], {row["driver_id"] for row in rows})
        for event_id, rows in history.by_event.items()
    }
    result = audit_strategy_history(
        root, events, progress=lambda message: print(message, flush=True)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
