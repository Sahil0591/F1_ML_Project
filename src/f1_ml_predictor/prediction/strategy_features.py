"""Tyre, pit-stop and long-run predictors from audited OpenF1 summaries.

History values use only races whose evidence was available by the cutoff, newest
first; practice values use this weekend's latest summarised session available by
the cutoff. Each value carries the clock of the evidence it came from.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

STRATEGY_WINDOW = 10
MIN_PRACTICE_DRIVERS = 6
HISTORY_FEATURES = (
    "driver_tyre_degradation_10",
    "constructor_tyre_degradation_10",
    "constructor_pit_lane_delta_10",
    "circuit_median_stops",
    "circuit_pit_lane_seconds",
)
PRACTICE_FEATURES = ("practice_long_run_gap", "practice_long_run_degradation")
STRATEGY_NUMERIC = (*HISTORY_FEATURES, *PRACTICE_FEATURES)


def _weighted_degradation(entries: list[list[float]]) -> float | None:
    if not entries:
        return None
    laps = sum(count for _, count in entries)
    return float(sum(value * count for value, count in entries) / laps)


def strategy_values(
    strategy: dict[str, dict[str, Any]],
    constructors: dict[str, dict[str, str]],
    circuits: dict[str, str],
    *,
    event_id: str,
    circuit_id: str,
    cutoff: datetime,
    roster: dict[str, str],
    practice: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Weekend-style values and clocks for build_rows, one entry per roster driver.

    ``constructors`` maps each past event to its driver constructors and
    ``circuits`` each past event to its circuit. ``practice`` overrides the
    audited practice summaries of this event (used live).
    """
    past = sorted(
        (
            (item["race_start"], key, item)
            for key, item in strategy.items()
            if key != event_id and item["available_at"] <= cutoff
        ),
        key=lambda value: value[0],
        reverse=True,
    )
    result: dict[str, dict[str, Any]] = {driver: {} for driver in roster}

    def put(driver: str, name: str, value: float | None, clock: datetime | None) -> None:
        if value is not None and clock is not None and np.isfinite(value):
            result[driver][name] = float(value)
            result[driver][f"{name}_available_at"] = clock

    for driver, constructor in roster.items():
        own: list[float] = []
        team: list[float] = []
        lanes: list[float] = []
        clocks: dict[str, datetime] = {}
        for _, key, item in past:
            drivers = item["race"]["drivers"]
            if len(own) < STRATEGY_WINDOW and driver in drivers:
                value = _weighted_degradation(drivers[driver]["degradation"])
                if value is not None:
                    own.append(value)
                    clocks.setdefault("own", item["available_at"])
            mates = [
                name
                for name, team_id in constructors.get(key, {}).items()
                if team_id == constructor and name in drivers
            ]
            if len(team) < STRATEGY_WINDOW:
                entries = [entry for name in mates for entry in drivers[name]["degradation"]]
                value = _weighted_degradation(entries)
                if value is not None:
                    team.append(value)
                    clocks.setdefault("team", item["available_at"])
            if len(lanes) < STRATEGY_WINDOW:
                deltas = [delta for name in mates for delta in drivers[name]["lane_deltas"]]
                if deltas:
                    lanes.append(float(np.mean(deltas)))
                    clocks.setdefault("lanes", item["available_at"])
        put(driver, "driver_tyre_degradation_10", np.mean(own) if own else None, clocks.get("own"))
        put(
            driver,
            "constructor_tyre_degradation_10",
            np.mean(team) if team else None,
            clocks.get("team"),
        )
        put(
            driver,
            "constructor_pit_lane_delta_10",
            np.mean(lanes) if lanes else None,
            clocks.get("lanes"),
        )
    same = [(key, item) for _, key, item in past if circuits.get(key) == circuit_id]
    stops = [
        item["race"]["median_stops"] for _, item in same if item["race"]["median_stops"] is not None
    ]
    lane = [
        item["race"]["lane_median"] for _, item in same if item["race"]["lane_median"] is not None
    ]
    clock = same[0][1]["available_at"] if same else None
    for driver in roster:
        put(driver, "circuit_median_stops", np.mean(stops) if stops else None, clock)
        put(driver, "circuit_pit_lane_seconds", np.mean(lane) if lane else None, clock)
    sessions = practice if practice is not None else strategy.get(event_id, {}).get("practice", [])
    usable = [
        session
        for session in sessions
        if session["available_at"] <= cutoff and len(session["drivers"]) >= MIN_PRACTICE_DRIVERS
    ]
    if usable:
        latest = max(usable, key=lambda session: session["available_at"])
        for driver in roster:
            values = latest["drivers"].get(driver, {})
            for name, key in (
                ("practice_long_run_gap", "long_run_gap"),
                ("practice_long_run_degradation", "long_run_degradation"),
            ):
                put(driver, name, values.get(key), latest["available_at"])
    return result
