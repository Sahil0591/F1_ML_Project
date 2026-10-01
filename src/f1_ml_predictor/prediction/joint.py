"""Coherent joint race distributions from latent utilities, with diagnostics.

Any strength model (linear, Plackett-Luce regression or boosting) supplies a
utility per driver. One sampler turns utilities into whole race orders:
independent DNF draws, Gumbel Plackett-Luce ordering at a calibrated temperature,
retirees trailing finishers. Calibration shrinkage is a mixture with the same
sampler at equal utilities (a DNF-aware uniform race prior), so every marginal
remains coherent. Optional per-draw offsets add persistent strength uncertainty.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

PODIUM = 3


def sample_orders(
    utility: np.ndarray[Any, Any],
    dnf: np.ndarray[Any, Any],
    *,
    temperature: float,
    draws: int,
    rng: np.random.Generator,
    shrink: float = 0.0,
    offsets: np.ndarray[Any, Any] | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return driver-index orders and DNF states for ``draws`` simulated races."""
    count = len(utility)
    if count == 0 or dnf.shape != utility.shape:
        raise ValueError("joint sampling needs equal nonempty driver vectors")
    if not np.isfinite(utility).all() or not np.isfinite(dnf).all():
        raise ValueError("joint sampling inputs must be finite")
    if (dnf < 0).any() or (dnf > 1).any() or not 0 <= shrink <= 1 or temperature <= 0:
        raise ValueError("invalid DNF probability, shrinkage or temperature")
    retired = rng.random((draws, count)) < dnf
    scaled = np.broadcast_to(utility / temperature, (draws, count))
    if offsets is not None:
        scaled = scaled + offsets
    prior = rng.random(draws) < shrink
    scaled = np.where(prior[:, None], 0.0, scaled)
    noisy = scaled + rng.gumbel(size=(draws, count))
    order = np.argsort(-noisy, axis=1, kind="stable")
    status = np.take_along_axis(retired, order, axis=1)
    order = np.take_along_axis(order, np.argsort(status, axis=1, kind="stable"), axis=1)
    return order, retired


def marginals(order: np.ndarray[Any, Any], retired: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Winner, podium, finish distribution, expected position and DNF from orders."""
    draws, count = order.shape
    positions = np.argsort(order, axis=1)
    finish = np.zeros((count, count))
    for driver in range(count):
        finish[driver] = np.bincount(positions[:, driver], minlength=count) / draws
    return {
        "winner": finish[:, 0].copy(),
        "podium": finish[:, : min(PODIUM, count)].sum(axis=1),
        "finish": finish,
        "expected": finish @ np.arange(1, count + 1),
        "dnf": retired.mean(axis=0),
    }


def mix(model: dict[str, Any], prior: dict[str, Any], shrink: float) -> dict[str, Any]:
    """Exact marginals of the mixture distribution."""
    return {
        key: (1 - shrink) * np.asarray(model[key]) + shrink * np.asarray(prior[key])
        for key in ("winner", "podium", "finish", "expected", "dnf")
    }


def average(components: list[dict[str, Any]]) -> dict[str, Any]:
    """Equal-weight mixture of coherent component distributions."""
    return {
        key: np.mean([np.asarray(item[key]) for item in components], axis=0)
        for key in ("winner", "podium", "finish", "expected", "dnf")
    }


def objective(values: dict[str, Any], winner: int, podium: np.ndarray[Any, Any]) -> float:
    """Calibration objective: winner log loss plus race podium log loss over three."""
    eps = 1e-12
    win = -math.log(max(float(values["winner"][winner]), eps))
    probability = np.clip(np.asarray(values["podium"], dtype=float), eps, 1 - eps)
    podium_loss = -np.sum(podium * np.log(probability) + (1 - podium) * np.log(1 - probability))
    return win + float(podium_loss) / PODIUM


def sharpness(winner: np.ndarray[Any, Any], podium: np.ndarray[Any, Any]) -> dict[str, float]:
    """Field-level sharpness statistics, independent of the outcome."""
    probability = np.asarray(winner, dtype=float)
    positive = probability[probability > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {
        "winner_entropy": entropy,
        "effective_win_contenders": math.exp(entropy),
        "fraction_below_0_1_percent_win": float((probability < 0.001).mean()),
        "fraction_below_1_percent_podium": float((np.asarray(podium) < 0.01).mean()),
        "maximum_win_probability": float(probability.max()),
    }


def expected_calibration_error(
    probability: list[float], outcome: list[bool], bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """Weighted absolute calibration gap across equal-width probability bins."""
    if not probability:
        return math.nan, []
    values = np.asarray(probability, dtype=float)
    truth = np.asarray(outcome, dtype=float)
    index = np.minimum((values * bins).astype(int), bins - 1)
    curve = []
    error = 0.0
    for bucket in range(bins):
        selected = index == bucket
        if not selected.any():
            continue
        predicted, observed = float(values[selected].mean()), float(truth[selected].mean())
        error += selected.mean() * abs(predicted - observed)
        curve.append(
            {
                "bin": bucket,
                "count": int(selected.sum()),
                "mean_predicted": predicted,
                "observed_rate": observed,
            }
        )
    return float(error), curve


def sample_mixture(
    components: list[tuple[np.ndarray[Any, Any], float, float]],
    dnf: np.ndarray[Any, Any],
    *,
    draws: int,
    rng: np.random.Generator,
    offsets: np.ndarray[Any, Any] | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Sample an equal-weight mixture of calibrated components draw by draw.

    Each component is ``(utility, temperature, shrink)``. A single component is
    the ordinary calibrated joint distribution.
    """
    count = len(dnf)
    choice = rng.integers(len(components), size=draws)
    order = np.empty((draws, count), dtype=np.intp)
    retired = np.empty((draws, count), dtype=bool)
    for index, (utility, temperature, shrink) in enumerate(components):
        selected = np.flatnonzero(choice == index)
        if not len(selected):
            continue
        part_order, part_retired = sample_orders(
            utility,
            dnf,
            temperature=temperature,
            draws=len(selected),
            rng=rng,
            shrink=shrink,
            offsets=None if offsets is None else offsets[selected],
        )
        order[selected] = part_order
        retired[selected] = part_retired
    return order, retired
