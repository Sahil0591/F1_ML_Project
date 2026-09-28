"""A seeded joint race-order model with coherent driver and position marginals."""

from typing import Any

import numpy as np


def race_distribution(
    position_scores: list[float],
    dnf_probability: list[float],
    *,
    temperature: float = 1.0,
    draws: int = 4096,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample PL orders, then place sampled retirements after sampled finishers.

    Finish means a total modeled order, including trailing retirements. It does
    not predict FIA classified status or the published ordinal of a retirement.
    """
    score = np.asarray(position_scores, dtype=float)
    dnf = np.asarray(dnf_probability, dtype=float)
    count = len(score)
    if count == 0 or dnf.shape != score.shape or score.ndim != 1:
        raise ValueError("distribution requires equal nonempty driver vectors")
    if not np.isfinite(score).all() or not np.isfinite(dnf).all():
        raise ValueError("distribution inputs must be finite")
    if (dnf < 0).any() or (dnf > 1).any():
        raise ValueError("DNF probabilities must lie in [0, 1]")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if isinstance(draws, bool) or not isinstance(draws, int) or not 128 <= draws <= 65536:
        raise ValueError("draws must be an integer from 128 to 65536")
    rng = np.random.default_rng(seed)
    retired = rng.random((draws, count)) < dnf
    utility = -score / temperature + rng.gumbel(size=(draws, count))
    pl_order = np.argsort(-utility, axis=1, kind="stable")
    statuses = np.take_along_axis(retired, pl_order, axis=1)
    order = np.take_along_axis(pl_order, np.argsort(statuses, axis=1, kind="stable"), axis=1)
    positions = np.argsort(order, axis=1)
    rows = []
    for driver in range(count):
        probabilities = np.bincount(positions[:, driver], minlength=count) / draws
        rows.append(
            {
                "winner_probability": float(probabilities[0]),
                "podium_probability": float(probabilities[: min(3, count)].sum()),
                "dnf_probability": float(retired[:, driver].mean()),
                "finish_distribution": probabilities.tolist(),
                "expected_position": float(probabilities @ np.arange(1, count + 1)),
                "simulation_standard_error_max": 0.5 / np.sqrt(draws),
            }
        )
    return rows
