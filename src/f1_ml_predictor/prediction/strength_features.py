"""Strength features adapted from public F1 predictors, computed point in time.

Three ideas are taken from open-source models and rebuilt on audited Gold history:

* Pairwise Elo (after Malek1414/f1-predictions): every pair of classified
  finishers is one match. The expected score uses driver plus constructor
  rating, so a fast car does not inflate its driver. Team-mate pairs count
  double for drivers and not at all for constructors, because the car cancels.
  Ratings regress towards the mean between seasons, constructors harder, and
  hardest in regulation-reset seasons.
* Team-mate qualifying head-to-head (after MynosIII/F1Predictor): the share of
  recent appearances in which a driver out-qualified the team-mate, which
  isolates driver pace from machinery.
* Circuit fingerprints (after MynosIII/F1Predictor): a small reference profile
  per circuit lets an unseen or rarely visited circuit borrow how a driver and
  a constructor ran, relative to their own baseline, at the most similar
  circuits. The profile table is static reference data, not an outcome, so it
  carries no point-in-time risk.

Only rows of earlier races whose labels were published before the cutoff are
passed in, and DNF labels are only used once their own clock precedes it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from statistics import mean
from typing import Any

ELO_BASE = 1500.0
ELO_SCALE = 400.0
DRIVER_K = 24.0
CONSTRUCTOR_K = 24.0
TEAMMATE_WEIGHT = 2.0
DRIVER_CARRYOVER = 0.75
CONSTRUCTOR_CARRYOVER = 0.5
RESET_CONSTRUCTOR_CARRYOVER = 0.2
# Seasons that began a new technical regulation cycle.
REGULATION_RESETS = frozenset({2014, 2017, 2022, 2026})
HEAD_TO_HEAD_WINDOW = 10
SIMILAR_CIRCUITS = 5
SIMILARITY_APPEARANCES = 40
SIMILARITY_BANDWIDTH = 1.0

# Hand-curated reference profile per circuit: lap length (km), number of turns,
# street or temporary circuit (1/0), downforce level (1 low, 2 medium, 3 high).
# Review and extend when the calendar changes; a circuit absent from this table
# leaves the similarity features missing rather than guessed.
CIRCUIT_PROFILES: dict[str, tuple[float, float, float, float]] = {
    "albert_park": (5.278, 14, 1, 2),
    "americas": (5.513, 20, 0, 2),
    "bahrain": (5.412, 15, 0, 2),
    "baku": (6.003, 20, 1, 1),
    "catalunya": (4.657, 14, 0, 3),
    "hungaroring": (4.381, 14, 0, 3),
    "imola": (4.909, 19, 0, 2),
    "interlagos": (4.309, 15, 0, 2),
    "jeddah": (6.174, 27, 1, 1),
    "losail": (5.419, 16, 0, 2),
    "madring": (5.474, 22, 1, 2),
    "marina_bay": (4.940, 19, 1, 3),
    "miami": (5.412, 19, 1, 2),
    "monaco": (3.337, 19, 1, 3),
    "monza": (5.793, 11, 0, 1),
    "red_bull_ring": (4.318, 10, 0, 2),
    "ricard": (5.842, 15, 0, 2),
    "rodriguez": (4.304, 17, 0, 3),
    "sepang": (5.543, 15, 0, 2),
    "shanghai": (5.451, 16, 0, 2),
    "silverstone": (5.891, 18, 0, 2),
    "spa": (7.004, 19, 0, 1),
    "suzuka": (5.807, 18, 0, 3),
    "vegas": (6.201, 17, 1, 1),
    "villeneuve": (4.361, 14, 1, 1),
    "yas_marina": (5.281, 16, 0, 2),
    "zandvoort": (4.259, 14, 0, 3),
}

STRENGTH_NUMERIC = (
    "driver_elo",
    "constructor_elo",
    "driver_teammate_qualifying_h2h_10",
    "driver_similar_circuit_delta",
    "constructor_similar_circuit_delta",
)
STRENGTH_COUNTS = ("driver_elo_events",)


def _season(event_id: str) -> int:
    return int(event_id.split("/")[0].removeprefix("season="))


def _classified(rows: Iterable[dict[str, Any]], cutoff: datetime) -> list[tuple[str, str, int]]:
    """(driver, constructor, position) for classified non-retired finishers."""
    result = []
    for row in rows:
        if row["label_position"] is None:
            continue
        retired = (
            row.get("label_dnf") is True
            and row.get("label_dnf_available_at") is not None
            and row["label_dnf_available_at"] < cutoff
        )
        if not retired:
            result.append((row["driver_id"], row["constructor_id"], int(row["label_position"])))
    return sorted(result, key=lambda item: item[2])


def _expected(gap: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, -gap / ELO_SCALE))


def elo_ratings(
    events: Sequence[Sequence[dict[str, Any]]], cutoff: datetime
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Driver ratings, constructor ratings and rated-event counts, oldest event first."""
    drivers: dict[str, float] = {}
    constructors: dict[str, float] = {}
    counts: dict[str, int] = {}
    season: int | None = None
    for rows in events:
        if not rows:
            continue
        current = _season(rows[0]["event_id"])
        if season is not None and current != season:
            carry = (
                RESET_CONSTRUCTOR_CARRYOVER
                if current in REGULATION_RESETS
                else CONSTRUCTOR_CARRYOVER
            )
            for name, value in drivers.items():
                drivers[name] = ELO_BASE + DRIVER_CARRYOVER * (value - ELO_BASE)
            for name, value in constructors.items():
                constructors[name] = ELO_BASE + carry * (value - ELO_BASE)
        season = current
        field = _classified(rows, cutoff)
        if len(field) < 2:
            continue
        strength = {
            driver: drivers.get(driver, ELO_BASE) + constructors.get(team, ELO_BASE) - ELO_BASE
            for driver, team, _ in field
        }
        driver_delta: dict[str, float] = dict.fromkeys(strength, 0.0)
        driver_weight: dict[str, float] = dict.fromkeys(strength, 0.0)
        team_delta: dict[str, float] = {}
        team_weight: dict[str, float] = {}
        for i, (first, first_team, _) in enumerate(field):
            for second, second_team, _ in field[i + 1 :]:
                # ``first`` finished ahead of ``second``.
                surprise = 1.0 - _expected(strength[first] - strength[second])
                same_team = first_team == second_team
                weight = TEAMMATE_WEIGHT if same_team else 1.0
                driver_delta[first] += weight * surprise
                driver_delta[second] -= weight * surprise
                driver_weight[first] += weight
                driver_weight[second] += weight
                if not same_team:
                    for team, sign in ((first_team, 1.0), (second_team, -1.0)):
                        team_delta[team] = team_delta.get(team, 0.0) + sign * surprise
                        team_weight[team] = team_weight.get(team, 0.0) + 1.0
        for driver, team, _ in field:
            drivers[driver] = drivers.get(driver, ELO_BASE) + (
                DRIVER_K * driver_delta[driver] / driver_weight[driver]
                if driver_weight[driver]
                else 0.0
            )
            counts[driver] = counts.get(driver, 0) + 1
            constructors.setdefault(team, ELO_BASE)
        for team, delta in team_delta.items():
            constructors[team] += CONSTRUCTOR_K * delta / team_weight[team]
    return drivers, constructors, counts


def teammate_head_to_head(
    appearances: Sequence[dict[str, Any]],
    events: Mapping[str, Sequence[dict[str, Any]]],
    window: int = HEAD_TO_HEAD_WINDOW,
) -> float | None:
    """Share of the latest ``window`` comparable appearances out-qualifying the team-mate."""
    outcomes: list[float] = []
    for item in appearances:
        own = item.get("qualifying_position")
        if own is None:
            continue
        mates = [
            other["qualifying_position"]
            for other in events[item["event_id"]]
            if other["constructor_id"] == item["constructor_id"]
            and other["driver_id"] != item["driver_id"]
            and other.get("qualifying_position") is not None
        ]
        if not mates:
            continue
        best_mate = min(mates)
        outcomes.append(1.0 if own < best_mate else 0.5 if own == best_mate else 0.0)
        if len(outcomes) == window:
            break
    return mean(outcomes) if outcomes else None


def _normalized_profiles() -> dict[str, tuple[float, ...]]:
    columns = list(zip(*CIRCUIT_PROFILES.values(), strict=True))
    centres = [mean(column) for column in columns]
    scales = [
        math.sqrt(mean((value - centre) ** 2 for value in column)) or 1.0
        for column, centre in zip(columns, centres, strict=True)
    ]
    return {
        name: tuple(
            (value - centre) / scale
            for value, centre, scale in zip(profile, centres, scales, strict=True)
        )
        for name, profile in CIRCUIT_PROFILES.items()
    }


_PROFILES = _normalized_profiles()


def circuit_similarity(first: str, second: str) -> float | None:
    """Gaussian similarity in standardized profile space, or None if unprofiled."""
    if first not in _PROFILES or second not in _PROFILES:
        return None
    distance = sum((a - b) ** 2 for a, b in zip(_PROFILES[first], _PROFILES[second], strict=True))
    return math.exp(-distance / (2 * SIMILARITY_BANDWIDTH**2))


def similar_circuits(circuit_id: str, count: int = SIMILAR_CIRCUITS) -> dict[str, float]:
    """The ``count`` most similar other profiled circuits with their weights."""
    scored = [
        (other, circuit_similarity(circuit_id, other)) for other in _PROFILES if other != circuit_id
    ]
    ranked = sorted(
        ((other, value) for other, value in scored if value is not None),
        key=lambda item: (-item[1], item[0]),
    )
    return dict(ranked[:count])


def similar_circuit_delta(
    results: Sequence[tuple[str, float]],
    circuit_id: str,
    limit: int = SIMILARITY_APPEARANCES,
) -> float | None:
    """Similarity-weighted finish at nearby circuits minus the overall mean.

    ``results`` holds ``(circuit_id, finish_position)`` pairs, latest first.
    Negative values mean better than usual at circuits like this one.
    """
    neighbours = similar_circuits(circuit_id)
    recent = list(results[:limit])
    if not neighbours or not recent:
        return None
    baseline = mean(position for _, position in recent)
    weighted = [
        (neighbours[circuit], position - baseline)
        for circuit, position in recent
        if circuit in neighbours
    ]
    total = sum(weight for weight, _ in weighted)
    if not total:
        return None
    return sum(weight * delta for weight, delta in weighted) / total
