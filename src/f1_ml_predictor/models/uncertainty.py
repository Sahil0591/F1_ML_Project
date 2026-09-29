"""Paired race-level uncertainty for candidate versus baseline losses."""

import math
from collections import defaultdict
from typing import Any

import numpy as np


def paired_loss_intervals(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    baseline_name: str,
    *,
    seed: int,
    resamples: int = 2000,
) -> dict[str, dict[str, Any]]:
    """Bootstrap paired event loss deltas, retaining whole driver rosters."""
    predictor = {
        "winner_log_loss": ("winner_probability", f"{baseline_name}_winner_probability"),
        "podium_brier": ("podium_probability", f"{baseline_name}_podium_probability"),
        "dnf_brier": ("dnf_probability", f"{baseline_name}_dnf_probability"),
        "position_mae": (
            "expected_position",
            "linear_finish_position" if baseline_name == "logistic" else "grid_finish_position",
        ),
    }
    paired = {
        (
            row["event_id"],
            row["prediction_timestamp"],
            row["cutoff_kind"],
            row["driver_id"],
        ): row
        for row in baseline
    }
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in candidate:
        key = (
            row["event_id"],
            row["prediction_timestamp"],
            row["cutoff_kind"],
            row["driver_id"],
        )
        if key not in paired:
            raise ValueError("paired uncertainty requires identical driver-race observations")
        if any(
            row[target] != paired[key][target]
            for target in ("label_winner", "label_podium", "label_dnf", "label_position")
        ):
            raise ValueError("paired uncertainty requires identical audited labels")
        groups[row["event_id"]].append((row, paired[key]))
    if len(paired) != len(candidate):
        raise ValueError("paired uncertainty requires identical driver-race observations")
    result: dict[str, dict[str, Any]] = {}
    for metric, (candidate_key, baseline_key) in predictor.items():
        deltas = []
        observations = 0
        for group in groups.values():
            values = []
            for row, reference in group:
                target_key = (
                    "label_winner"
                    if metric == "winner_log_loss"
                    else "label_podium"
                    if metric == "podium_brier"
                    else "label_dnf"
                    if metric == "dnf_brier"
                    else "label_position"
                )
                target = row[target_key]
                if target is None or (metric == "winner_log_loss" and not target):
                    continue
                actual = float(row[candidate_key])
                base = float(reference[baseline_key])
                if metric == "winner_log_loss":
                    values.append(-math.log(max(actual, 1e-15)) + math.log(max(base, 1e-15)))
                elif metric.endswith("brier"):
                    truth = float(target)
                    values.append((actual - truth) ** 2 - (base - truth) ** 2)
                else:
                    values.append(abs(actual - float(target)) - abs(base - float(target)))
            if values:
                deltas.append(sum(values) / len(values))
                observations += len(values)
        if len(deltas) < 2:
            result[metric] = {"status": "insufficient_data", "paired_events": len(deltas)}
            continue
        values_array = np.asarray(deltas)
        generator = np.random.default_rng(seed)
        sampled = generator.choice(values_array, size=(resamples, len(deltas)), replace=True)
        lower, upper = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
        result[metric] = {
            "status": "estimated",
            "paired_events": len(deltas),
            "driver_race_observations": observations,
            "mean_loss_delta": float(values_array.mean()),
            "bootstrap_95_percent_interval": [float(lower), float(upper)],
            "unit": "race",
            "resamples": resamples,
            "seed": seed,
        }
    return result
