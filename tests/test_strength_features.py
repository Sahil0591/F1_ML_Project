from datetime import UTC, datetime, timedelta

import pytest

from f1_ml_predictor.prediction.strength_features import (
    CIRCUIT_PROFILES,
    ELO_BASE,
    circuit_similarity,
    elo_ratings,
    similar_circuit_delta,
    similar_circuits,
    teammate_head_to_head,
)

CUTOFF = datetime(2026, 10, 1, tzinfo=UTC)


def _row(event: str, driver: str, team: str, position: int | None, **extra: object) -> dict:
    return {
        "event_id": event,
        "driver_id": driver,
        "constructor_id": team,
        "label_position": position,
        "label_dnf": None,
        "label_dnf_available_at": None,
        **extra,
    }


def _race(event: str, order: list[tuple[str, str]]) -> list[dict]:
    return [_row(event, driver, team, place) for place, (driver, team) in enumerate(order, 1)]


def test_elo_rewards_winners_and_separates_driver_from_car() -> None:
    order = [("a1", "fast"), ("a2", "fast"), ("b1", "slow"), ("b2", "slow")]
    events = [_race(f"season=2025/round={n:02d}", order) for n in range(1, 9)]
    drivers, constructors, counts = elo_ratings(events, CUTOFF)
    assert constructors["fast"] > ELO_BASE > constructors["slow"]
    assert drivers["a1"] > drivers["a2"]
    assert drivers["b1"] > drivers["b2"]
    assert counts == {"a1": 8, "a2": 8, "b1": 8, "b2": 8}


def test_teammate_battles_do_not_move_constructor_ratings() -> None:
    events = [_race("season=2025/round=01", [("a1", "team"), ("a2", "team")])]
    drivers, constructors, _ = elo_ratings(events, CUTOFF)
    assert constructors["team"] == ELO_BASE
    assert drivers["a1"] > ELO_BASE > drivers["a2"]


def test_ratings_regress_between_seasons_harder_at_regulation_resets() -> None:
    order = [("a1", "fast"), ("b1", "slow")]
    season = [_race(f"season=2024/round={n:02d}", order) for n in range(1, 6)]
    _, before, _ = elo_ratings(season, CUTOFF)
    _, normal, _ = elo_ratings([*season, []], CUTOFF)
    tiny = [_row("season=2025/round=01", "x", "other", 1)]
    _, carried, _ = elo_ratings([*season, tiny], CUTOFF)
    reset = [_row("season=2026/round=01", "x", "other", 1)]
    _, after_reset, _ = elo_ratings([*season, reset], CUTOFF)
    assert normal == before
    gain = before["fast"] - ELO_BASE
    assert carried["fast"] - ELO_BASE == pytest.approx(0.5 * gain)
    assert after_reset["fast"] - ELO_BASE == pytest.approx(0.2 * gain)


def test_retirements_count_only_once_their_label_is_published() -> None:
    race = _race("season=2025/round=01", [("a1", "x"), ("b1", "y")])
    race[0].update(label_dnf=True, label_dnf_available_at=CUTOFF + timedelta(days=1))
    drivers, _, _ = elo_ratings([race], CUTOFF)
    assert drivers["a1"] > ELO_BASE
    race[0]["label_dnf_available_at"] = CUTOFF - timedelta(days=1)
    drivers, _, counts = elo_ratings([race], CUTOFF)
    assert drivers == {} and counts == {}


def test_teammate_head_to_head_uses_latest_window() -> None:
    events = {}
    appearances = []
    for n in range(1, 13):
        event = f"season=2025/round={n:02d}"
        ahead = n > 2
        own = _row(event, "a1", "t", 1, qualifying_position=1.0 if ahead else 2.0)
        mate = _row(event, "a2", "t", 2, qualifying_position=2.0 if ahead else 1.0)
        events[event] = [own, mate]
        appearances.insert(0, own)
    assert teammate_head_to_head(appearances, events) == 1.0
    assert teammate_head_to_head(appearances, events, window=12) == pytest.approx(10 / 12)
    assert teammate_head_to_head([], events) is None


def test_circuit_profiles_cover_every_scheduled_venue_and_sepang() -> None:
    assert {"sepang", "bahrain", "madring", "catalunya"} <= set(CIRCUIT_PROFILES)
    assert circuit_similarity("sepang", "sepang") == pytest.approx(1.0)
    assert circuit_similarity("sepang", "unknown") is None
    neighbours = similar_circuits("sepang")
    assert len(neighbours) == 5 and "sepang" not in neighbours
    assert circuit_similarity("sepang", "bahrain") > circuit_similarity("sepang", "monaco")


def test_similar_circuit_delta_is_relative_to_own_baseline() -> None:
    near = next(iter(similar_circuits("sepang")))
    results = [(near, 1.0), ("monaco", 9.0), (near, 1.0), ("monaco", 9.0)]
    assert similar_circuit_delta(results, "sepang") == pytest.approx(-4.0)
    assert similar_circuit_delta(results, "nowhere") is None
    assert similar_circuit_delta([], "sepang") is None
