"""Windowless Windows task entry point with a durable local tick log."""

import argparse
import json
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from f1_ml_predictor.trust.scheduler import scheduler_tick


def run_task(root: Path, *, tick: Callable[[Path], dict[str, Any]] = scheduler_tick) -> int:
    root = root.resolve()
    observed = datetime.now(UTC)
    try:
        state = tick(root)
        record = {
            "observed_at_utc": observed.isoformat(),
            "status": state["status"],
            "error": state.get("error"),
            "missed_event_ids": state.get("missed_event_ids", []),
            "deferred_outcomes": {
                name: entry["outcome_collection"].get("reason")
                for name, entry in state.get("events", {}).items()
                if entry.get("outcome_collection", {}).get("status") == "deferred"
            },
            "next_check_at": state.get("next_check_at"),
        }
        exit_code = 1 if state["status"] == "error" else 0
    except Exception as exc:
        record = {
            "observed_at_utc": observed.isoformat(),
            "status": "exception",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    directory = root / "data/raw/prospective_scheduler/logs"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"{observed.date().isoformat()}.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_task(args.root))


if __name__ == "__main__":
    main()
