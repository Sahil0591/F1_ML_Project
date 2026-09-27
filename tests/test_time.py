from datetime import UTC, datetime, timedelta, timezone

import pytest

from f1_ml_predictor.time import require_known_by


def test_input_available_at_cutoff_is_allowed() -> None:
    cutoff = datetime(2026, 9, 27, 12, tzinfo=UTC)
    require_known_by(cutoff, cutoff)
    require_known_by(cutoff - timedelta(seconds=1), cutoff)


def test_future_input_is_rejected() -> None:
    cutoff = datetime(2026, 9, 27, 12, tzinfo=UTC)
    with pytest.raises(ValueError, match="after prediction_timestamp"):
        require_known_by(cutoff + timedelta(seconds=1), cutoff)


@pytest.mark.parametrize("which", ["available", "prediction"])
def test_naive_timestamps_are_rejected(which: str) -> None:
    aware = datetime(2026, 9, 27, 12, tzinfo=UTC)
    naive = aware.replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        require_known_by(
            naive if which == "available" else aware, naive if which == "prediction" else aware
        )


def test_non_utc_offset_is_rejected() -> None:
    cutoff = datetime(2026, 9, 27, 12, tzinfo=UTC)
    local = datetime(2026, 9, 27, 13, tzinfo=timezone(timedelta(hours=1)))
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        require_known_by(local, cutoff)
