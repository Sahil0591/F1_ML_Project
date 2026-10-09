import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from f1_ml_predictor.trust import practice_capture
from f1_ml_predictor.trust.practice_capture import captured_practice_values

_PARSED = {
    "russell": {"position": 1, "best_lap_seconds": 92.274},
    "antonelli": {"position": 3, "best_lap_seconds": 92.774},
    "max_verstappen": {"position": 2, "best_lap_seconds": 92.5},
    "hadjar": {"position": 4, "best_lap_seconds": None},
}
_ROSTER = {
    "russell": "mercedes",
    "antonelli": "mercedes",
    "max_verstappen": "red_bull",
    "hadjar": "red_bull",
    "albon": "williams",
}
_CUTOFF = datetime(2026, 10, 9, 16, tzinfo=UTC)


def _record(tmp_path: Path, available_at: str, field_size: int = 22) -> dict:
    pdf = tmp_path / "practice.pdf"
    pdf.write_bytes(b"%PDF retained")
    return {
        "available_at": available_at,
        "session": 1,
        "field_size": field_size,
        "document": {
            "document_id": "16",
            "url": "https://www.fia.com/practice.pdf",
            "path": "practice.pdf",
            "sha256": hashlib.sha256(b"%PDF retained").hexdigest(),
        },
        "bundle": {"path": "capture.json", "sha256": "x"},
    }


@pytest.fixture(autouse=True)
def _parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    def parse(path: Path, expected: set[str]) -> tuple[dict, int]:
        return {driver: row for driver, row in _PARSED.items() if driver in expected}, 22

    monkeypatch.setattr(practice_capture, "parse_practice_pdf", parse)


def test_practice_published_by_cutoff_feeds_weekend_values(tmp_path: Path) -> None:
    values, source = captured_practice_values(
        tmp_path, _record(tmp_path, "2026-10-09T11:39:00+00:00"), _ROSTER, _CUTOFF
    )
    assert source is not None and source["session"] == 1
    russell = values["russell"]
    assert russell["practice_position"] == 1.0
    assert russell["session_relative_rank"] == 0.0
    assert russell["best_lap_gap_to_fastest"] == 0.0
    assert russell["teammate_practice_delta"] == pytest.approx(-0.5)
    assert russell["practice_position_available_at"] == datetime(2026, 10, 9, 11, 39, tzinfo=UTC)
    assert values["hadjar"]["best_lap_gap_to_fastest"] is None
    assert values["max_verstappen"]["teammate_practice_delta"] is None
    assert values["albon"]["practice_position"] is None


def test_practice_published_after_cutoff_is_not_used(tmp_path: Path) -> None:
    record = _record(tmp_path, "2026-10-09T16:00:01+00:00")
    assert captured_practice_values(tmp_path, record, _ROSTER, _CUTOFF) == ({}, None)
    assert captured_practice_values(tmp_path, None, _ROSTER, _CUTOFF) == ({}, None)


def test_changed_practice_pdf_or_field_is_rejected(tmp_path: Path) -> None:
    record = _record(tmp_path, "2026-10-09T11:39:00+00:00")
    (tmp_path / "practice.pdf").write_bytes(b"%PDF changed")
    with pytest.raises(ValueError, match="changed"):
        captured_practice_values(tmp_path, record, _ROSTER, _CUTOFF)
    record = _record(tmp_path, "2026-10-09T11:39:00+00:00", field_size=20)
    with pytest.raises(ValueError, match="field size"):
        captured_practice_values(tmp_path, record, _ROSTER, _CUTOFF)
