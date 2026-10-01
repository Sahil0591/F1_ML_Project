"""Validated form drift for season simulation (protocol addendum season-drift-v1).

A season simulation predicts races several rounds ahead with form frozen at the
cutoff. For each historical cutoff race, this rebuilds pre-weekend rows for the
following races of the same season using only information known at that cutoff
(form frozen, circuit and format specific), scores them with the model fitted
for that cutoff, and finds the per-race random-walk variance of driver strength
that keeps those stale forecasts calibrated. Lag zero is the cutoff race itself.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.prediction.candidates import fit_dnf, fit_strength
from f1_ml_predictor.prediction.contracts import AuditedHistory, build_rows
from f1_ml_predictor.prediction.evaluation import columns_for, training_rows
from f1_ml_predictor.prediction.joint import marginals, objective, sample_orders
from f1_ml_predictor.prediction.protocol import CALIBRATION_DRAWS, MIN_CALIBRATION_EVENTS

DRIFT_VERSION = "season-drift-v1"
DRIFT_GRID = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 1.5, 2.25)
HORIZON = 8
DRIFT_PROTOCOL = {
    "version": DRIFT_VERSION,
    "contract": "pre_weekend",
    "horizon_races": HORIZON,
    "per_race_variance_grid": list(DRIFT_GRID),
    "strength_model": "best single candidate of the cutoff evaluation",
    "calibration": "cutoff race temperature and shrinkage held fixed",
    "objective": "winner log loss plus race podium log loss divided by three",
    "draws": CALIBRATION_DRAWS,
}
DRIFT_SHA256 = hashlib.sha256(
    json.dumps(DRIFT_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _key(event_id: str) -> EventId:
    season, number = event_id.split("/")
    return EventId(int(season.removeprefix("season=")), int(number.removeprefix("round=")))


def estimate_form_drift(
    root: Path,
    history: AuditedHistory,
    rows: list[dict[str, Any]],
    evaluation: dict[str, Any],
    *,
    masked: tuple[str, ...],
    seed: int,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Cached estimate of per-race strength drift for stale-form forecasts."""
    live = evaluation["live"]
    name = live["strength_model_for_uncertainty"]
    payload = {
        "dataset_sha256": dataset_sha256,
        "masked_features": sorted(masked),
        "strength_model": name,
        "seed": seed,
        "drift_protocol_sha256": DRIFT_SHA256,
        "evaluation_protocol_sha256": evaluation["protocol_sha256"],
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = (
        root
        / "models/experiments/gold"
        / history.version.dataset_version
        / "cutoff_v3/pre_weekend/drift"
        / f"{key}.json"
    )
    if path.exists():
        cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cached
    columns, dnf_columns = columns_for("pre_weekend", masked)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["event_id"], []).append(row)
    order = sorted(grouped, key=lambda event: (grouped[event][0]["prediction_timestamp"], event))
    parameters = {item["event_id"]: item for item in evaluation["calibration_parameters"][name]}
    pairs: list[tuple[int, list[dict[str, Any]], np.ndarray[Any, Any], Any, float, float, int]] = []
    for position, event_id in enumerate(order):
        if event_id not in parameters:
            continue
        calibration = parameters[event_id]
        if calibration["calibration_events"] < MIN_CALIBRATION_EVENTS:
            continue
        cutoff: datetime = grouped[event_id][0]["prediction_timestamp"]
        train, _ = training_rows(rows, cutoff)
        model = fit_strength(name, train, columns, seed=seed)
        dnf_model = fit_dnf(live["dnf_model"], train, dnf_columns, seed)
        season = event_id[7:11]
        for lag in range(HORIZON + 1):
            if position + lag >= len(order) or order[position + lag][7:11] != season:
                break
            target = order[position + lag]
            labelled = sorted(grouped[target], key=lambda row: row["driver_id"])
            roster = {row["driver_id"]: row["constructor_id"] for row in labelled}
            stale, _ = build_rows(
                history,
                "pre_weekend",
                event=_key(target),
                circuit_id=labelled[0]["circuit_id"],
                cutoff=cutoff,
                roster=roster,
            )
            winner = next(index for index, row in enumerate(labelled) if row["label_winner"])
            podium = np.asarray([float(row["label_podium"]) for row in labelled])
            pairs.append(
                (
                    lag,
                    stale,
                    model.utility(stale),
                    (winner, podium, dnf_model.predict(stale)),
                    calibration["temperature"],
                    calibration["shrink"],
                    position,
                )
            )
    scores: dict[float, dict[int, list[float]]] = {q: {} for q in DRIFT_GRID}
    for index, (lag, stale, utility, outcome, tau, shrink, _) in enumerate(pairs):
        winner, podium, dnf = outcome
        for variance in DRIFT_GRID:
            rng = np.random.default_rng((seed, 5, index))
            spread = np.sqrt(variance * lag)
            offsets = (
                rng.normal(0.0, spread, size=(CALIBRATION_DRAWS, len(stale)))
                if spread > 0
                else None
            )
            sampled = sample_orders(
                utility,
                dnf,
                temperature=tau,
                draws=CALIBRATION_DRAWS,
                rng=rng,
                shrink=shrink,
                offsets=offsets,
            )
            scores[variance].setdefault(lag, []).append(
                objective(marginals(*sampled), winner, podium)
            )
    by_variance = {
        variance: float(
            np.mean([value for lag, values in table.items() if lag > 0 for value in values])
        )
        for variance, table in scores.items()
    }
    best = min(DRIFT_GRID, key=lambda variance: (by_variance[variance], variance))
    by_lag = {
        str(lag): {
            "pairs": len(scores[0.0][lag]),
            "objective_without_drift": float(np.mean(scores[0.0][lag])),
            "objective_with_selected_drift": float(np.mean(scores[best][lag])),
        }
        for lag in sorted(scores[0.0])
    }
    result = {
        "version": DRIFT_VERSION,
        "protocol": DRIFT_PROTOCOL,
        "protocol_sha256": DRIFT_SHA256,
        "key": payload,
        "strength_model": name,
        "pairs": len(pairs),
        "cutoff_races": len({item[6] for item in pairs}),
        "objective_by_variance": {str(key): value for key, value in by_variance.items()},
        "per_race_variance": best,
        "grid_edge": best == DRIFT_GRID[-1],
        "by_lag": by_lag,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode())
    return result
