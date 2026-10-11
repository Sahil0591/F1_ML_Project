"""FIA starting grid parsing and race grid document selection."""

from pathlib import Path

import pytest

from f1_ml_predictor.trust import grid_history
from f1_ml_predictor.trust.grid_history import parse_grid_pdf, race_grid_records


class _Page:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


def _parse(monkeypatch: pytest.MonkeyPatch, text: str, roster: set[str]) -> dict:
    monkeypatch.setattr(
        grid_history, "PdfReader", lambda _: type("R", (), {"pages": [_Page(text)]})
    )
    return parse_grid_pdf(Path("grid.pdf"), roster)


ROSTER = {
    "max_verstappen": ("1", "Max VERSTAPPEN"),
    "leclerc": ("16", "Charles LECLERC"),
    "sainz": ("55", "Carlos SAINZ"),
    "perez": ("11", "Sergio PEREZ"),
    "russell": ("63", "George RUSSELL"),
    "hamilton": ("44", "Lewis HAMILTON"),
    "norris": ("4", "Lando NORRIS"),
    "ricciardo": ("3", "Daniel RICCIARDO"),
    "alonso": ("14", "Fernando ALONSO"),
    "ocon": ("31", "Esteban OCON"),
    "bottas": ("77", "Valtteri BOTTAS"),
    "gasly": ("10", "Pierre GASLY"),
    "tsunoda": ("22", "Yuki TSUNODA"),
    "vettel": ("5", "Sebastian VETTEL"),
    "stroll": ("18", "Lance STROLL"),
    "albon": ("23", "Alexander ALBON"),
    "latifi": ("6", "Nicholas LATIFI"),
    "magnussen": ("20", "Kevin MAGNUSSEN"),
    "mick_schumacher": ("47", "Mick SCHUMACHER"),
    "zhou": ("24", "ZHOU Guanyu"),
}


def test_race_grid_records_skip_stewards_and_sprint_documents() -> None:
    titles = [
        "Provisional Starting Grid",
        "Provisional Starting Grid (Corrected)",
        "Final Starting Grid",
        "Revised Final Starting Grid",
        "Offence - Car 6 - Starting Grid Infringement",
        "Provisional Sprint Starting Grid",
        "Final Sprint Starting Grid",
    ]
    kept = race_grid_records([{"title": title} for title in titles])
    assert [doc["title"] for doc in kept] == titles[:4]


def test_stacked_cells_are_rejoined_into_grid_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = []
    for place, (number, name) in enumerate(ROSTER.values(), 1):
        lines += [str(place), number, name + (" *" if place == 20 else ""), "Team", "1:30.000"]
    grid = _parse(monkeypatch, "\n".join(["Provisional Starting Grid", *lines]), set(ROSTER))
    assert grid["max_verstappen"] == {"grid_position": 1, "pit_lane_start": False}
    assert grid["zhou"] == {"grid_position": 20, "pit_lane_start": False}


def test_driver_under_pit_lane_heading_is_a_pit_lane_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        f"{place} {number} {name}"
        for place, (number, name) in enumerate(list(ROSTER.values())[:19], 1)
    ]
    text = "\n".join(
        [
            *rows,
            "DRIVERS REQUIRED TO START FROM THE PIT LANE",
            "24 ZHOU Guanyu *",
            "Alfa Romeo F1 Team ORLEN",
            "* PENALTIES",
        ]
    )
    grid = _parse(monkeypatch, text, set(ROSTER))
    assert grid["zhou"] == {"grid_position": None, "pit_lane_start": True}
    assert grid["mick_schumacher"]["grid_position"] == 19


def test_unnamed_omission_is_still_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        f"{place} {number} {name}"
        for place, (number, name) in enumerate(list(ROSTER.values())[:19], 1)
    ]
    with pytest.raises(ValueError, match="pit-lane statement"):
        _parse(
            monkeypatch,
            "\n".join([*rows, "DRIVERS REQUIRED TO START FROM THE PIT LANE"]),
            set(ROSTER),
        )
