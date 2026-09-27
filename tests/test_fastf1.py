"""Offline coverage for optional FastF1 lap ingestion."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.sources.fastf1 import load_session_summary, summarize_laps

EVENT = EventId(2025, 1)
CAPTURED = datetime(2025, 3, 15, tzinfo=UTC)
DRIVERS = {"1": "max_verstappen", "44": "hamilton"}


def lap(**changes: Any) -> dict[str, Any]:
    return {
        "DriverNumber": "1",
        "IsAccurate": True,
        "Deleted": False,
        "PitInTime": None,
        "PitOutTime": None,
        "LapTime": timedelta(seconds=90),
        "Compound": "SOFT",
        "TyreLife": 3.0,
        **changes,
    }


def test_summary_groups_driver_and_compound() -> None:
    rows = summarize_laps(
        [
            lap(LapTime=timedelta(seconds=92), TyreLife=5),
            lap(),
            lap(Compound="MEDIUM", LapTime=timedelta(seconds=93), TyreLife=None),
            lap(DriverNumber="44", LapTime=timedelta(seconds=89)),
        ],
        EVENT,
        "FP2",
        DRIVERS,
        CAPTURED,
    )
    assert len(rows) == 3
    soft = next(
        row for row in rows if row["driver_id"] == "max_verstappen" and row["compound"] == "SOFT"
    )
    assert soft == {
        "event_id": EVENT.partition(),
        "driver_id": "max_verstappen",
        "session_code": "FP2",
        "compound": "SOFT",
        "lap_count": 2,
        "median_lap_seconds": 91.0,
        "best_lap_seconds": 90.0,
        "median_tyre_age": 4.0,
        "available_at": CAPTURED,
    }
    assert next(row for row in rows if row["compound"] == "MEDIUM")["median_tyre_age"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"IsAccurate": False},
        {"IsAccurate": None},
        {"Deleted": True},
        {"Deleted": None},
        {"PitInTime": timedelta(seconds=100)},
        {"PitOutTime": timedelta(0)},
        {"LapTime": None},
        {"LapTime": 90},
        {"LapTime": timedelta(0)},
        {"LapTime": timedelta(seconds=-1)},
        {"Compound": None},
    ],
)
def test_invalid_laps_are_excluded(changes: dict[str, Any]) -> None:
    assert summarize_laps([lap(**changes)], EVENT, "Q", DRIVERS, CAPTURED) == []


def test_unknown_driver_is_an_explicit_error() -> None:
    with pytest.raises(ValueError, match="mapping.*'99'"):
        summarize_laps([lap(DriverNumber="99", IsAccurate=False)], EVENT, "Q", DRIVERS, CAPTURED)


def test_unknown_deletion_status_is_not_assumed_valid() -> None:
    record = lap()
    del record["Deleted"]
    assert summarize_laps([record], EVENT, "Q", DRIVERS, CAPTURED) == []


def test_nan_pit_fields_and_tyre_age() -> None:
    row = summarize_laps(
        [lap(PitInTime=float("nan"), PitOutTime=float("nan"), TyreLife=float("nan"))],
        EVENT,
        "FP1",
        DRIVERS,
        CAPTURED,
    )[0]
    assert row["median_tyre_age"] is None


def test_summary_rejects_race_and_non_utc_capture() -> None:
    with pytest.raises(ValueError, match="session_code"):
        summarize_laps([], EVENT, "R", DRIVERS, CAPTURED)
    with pytest.raises(ValueError, match="UTC"):
        summarize_laps([], EVENT, "Q", DRIVERS, CAPTURED.replace(tzinfo=None))


def test_load_uses_cache_and_only_lap_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = Mock()
    session = module.get_session.return_value
    session.results = None
    session.laps.to_dict.return_value = [lap()]
    monkeypatch.setattr("f1_ml_predictor.sources.fastf1.importlib.import_module", lambda _: module)
    before = datetime.now(UTC)
    rows = load_session_summary(EVENT, "FP3", DRIVERS, tmp_path / "cache")
    assert (tmp_path / "cache").is_dir()
    module.Cache.enable_cache.assert_called_once_with(str(tmp_path / "cache"))
    module.get_session.assert_called_once_with(2025, 1, "FP3")
    session.load.assert_called_once_with(laps=True, telemetry=False, weather=False, messages=True)
    session.laps.to_dict.assert_called_once_with(orient="records")
    assert before <= rows[0]["available_at"] <= datetime.now(UTC)


def test_loader_resolves_reserve_driver_source_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = Mock()
    session = module.get_session.return_value
    session.laps.to_dict.return_value = [lap(DriverNumber="99")]
    session.results.to_dict.return_value = [{"DriverNumber": "99", "DriverId": "reserve_driver"}]
    monkeypatch.setattr("f1_ml_predictor.sources.fastf1.importlib.import_module", lambda _: module)
    crosswalk = DRIVERS.copy()
    rows = load_session_summary(EVENT, "FP1", crosswalk, tmp_path)
    assert rows[0]["driver_id"] == "reserve_driver"
    assert crosswalk == DRIVERS
    session.results.to_dict.assert_called_once_with(orient="records")


@pytest.mark.parametrize("source_id", ["other_driver", "Invalid ID"])
def test_loader_rejects_conflicting_or_invalid_source_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_id: str,
) -> None:
    module = Mock()
    session = module.get_session.return_value
    session.laps.to_dict.return_value = [lap()]
    session.results.to_dict.return_value = [{"DriverNumber": "1", "DriverId": source_id}]
    monkeypatch.setattr("f1_ml_predictor.sources.fastf1.importlib.import_module", lambda _: module)
    with pytest.raises(ValueError, match="Conflicting|canonical Jolpica"):
        load_session_summary(EVENT, "FP1", DRIVERS, tmp_path)


@pytest.mark.parametrize(
    "result_records",
    [
        None,
        [],
        [{"DriverNumber": "99"}],
        [{"DriverNumber": "99", "DriverId": "", "FullName": "Reserve Driver"}],
    ],
)
def test_loader_does_not_join_missing_driver_ids_by_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result_records: list[dict[str, Any]] | None,
) -> None:
    module = Mock()
    session = module.get_session.return_value
    session.laps.to_dict.return_value = [lap(DriverNumber="99")]
    if result_records is None:
        session.results = None
    else:
        session.results.to_dict.return_value = result_records
    monkeypatch.setattr("f1_ml_predictor.sources.fastf1.importlib.import_module", lambda _: module)
    with pytest.raises(ValueError, match="mapping.*'99'"):
        load_session_summary(EVENT, "FP1", DRIVERS, tmp_path)


def test_missing_optional_dependency_has_install_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing(_: str) -> None:
        raise ModuleNotFoundError("No module named fastf1", name="fastf1")

    monkeypatch.setattr("f1_ml_predictor.sources.fastf1.importlib.import_module", missing)
    with pytest.raises(RuntimeError, match=r"f1-ml-predictor\[fastf1\]"):
        load_session_summary(EVENT, "Q", DRIVERS, tmp_path)
