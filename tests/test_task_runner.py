import json
from pathlib import Path

from f1_ml_predictor.trust.task_runner import run_task


def test_windowless_task_logs_status_and_failure(tmp_path: Path) -> None:
    assert run_task(tmp_path, tick=lambda _: {"status": "captured", "missed_event_ids": []}) == 0
    assert run_task(tmp_path, tick=lambda _: {"status": "error", "error": "source timeout"}) == 1
    logs = list((tmp_path / "data/raw/prospective_scheduler/logs").glob("*.jsonl"))
    assert len(logs) == 1
    records = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert [row["status"] for row in records] == ["captured", "error"]
    assert records[1]["error"] == "source timeout"
