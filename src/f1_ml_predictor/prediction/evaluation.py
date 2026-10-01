"""Cutoff-specific chronological evaluation, calibration and selection (protocol v3).

Stage one fits every candidate on complete earlier races for each outer race and
stores raw out-of-fold utilities. Stage two calibrates prequentially: a race's
temperature and prior shrinkage come only from earlier out-of-fold races. Model
choice is prequential too, so the selected pipeline has its own honest out-of-fold
score. Formal selection keeps the frozen v2 thresholds against both baselines.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from f1_ml_predictor.models.protocol import SELECTION_PAIRED_EVENTS
from f1_ml_predictor.prediction.candidates import (
    baseline_marginals,
    fit_dnf,
    fit_strength,
)
from f1_ml_predictor.prediction.contracts import dnf_columns, feature_columns
from f1_ml_predictor.prediction.joint import (
    average,
    expected_calibration_error,
    marginals,
    mix,
    objective,
    sample_orders,
    sharpness,
)
from f1_ml_predictor.prediction.protocol import (
    CALIBRATION_DRAWS,
    DEFAULT_CANDIDATE,
    DNF_CANDIDATES,
    JOINT_CANDIDATES,
    MIN_CALIBRATION_EVENTS,
    MIN_SELECTION_EVENTS,
    MIN_TRAIN_EVENTS,
    MIN_UNSEEN_CALIBRATION_EVENTS,
    PERSISTENT_SIGMAS,
    PROTOCOL_V3,
    PROTOCOL_V3_SHA256,
    SHRINKAGE,
    TEMPERATURES,
)

ENSEMBLE_MEMBERS = {
    "ensemble_linear": ("logistic_pl", "ridge_pl", "pl_regression"),
    "ensemble_all": JOINT_CANDIDATES,
}
BOOTSTRAP_RESAMPLES = 2000


def columns_for(contract: str, masked: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Contract predictors with unavailable features removed, with their flags."""
    hidden = set(masked)

    def keep(name: str) -> bool:
        return name not in hidden and name.removesuffix("_missing") not in hidden

    return (
        tuple(name for name in feature_columns(contract) if keep(name)),
        tuple(name for name in dnf_columns(contract) if keep(name)),
    )


@dataclass
class RaceRecord:
    event_id: str
    cutoff: datetime
    drivers: list[str]
    winner: int
    podium: np.ndarray[Any, Any]
    positions: np.ndarray[Any, Any]
    dnf_labels: list[bool | None]
    circuit_seen: bool
    train_events: int
    utilities: dict[str, np.ndarray[Any, Any]] = field(default_factory=dict)
    dnf: dict[str, np.ndarray[Any, Any]] = field(default_factory=dict)
    baselines: dict[str, dict[str, np.ndarray[Any, Any]]] = field(default_factory=dict)
    devices: dict[str, str] = field(default_factory=dict)


def training_rows(rows: list[dict[str, Any]], cutoff: datetime) -> tuple[list[dict[str, Any]], int]:
    """Complete earlier races with every label known by the cutoff; DNF has its own clock."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["event_id"], []).append(row)
    selected = []
    events = 0
    for group in grouped.values():
        if all(
            row["prediction_timestamp"] < cutoff and row["label_available_at"] <= cutoff
            for row in group
        ):
            events += 1
            for row in group:
                known = row["label_dnf_available_at"] is not None and (
                    row["label_dnf_available_at"] <= cutoff
                )
                selected.append({**row, "label_dnf": row["label_dnf"] if known else None})
    return selected, events


def collect_out_of_fold(
    rows: list[dict[str, Any]],
    contract: str,
    *,
    masked: tuple[str, ...] = (),
    seed: int = 42,
    device: str = "cpu",
    hardware: dict[str, Any] | None = None,
    candidates: tuple[str, ...] = JOINT_CANDIDATES,
    progress: Callable[[str], None] | None = None,
) -> list[RaceRecord]:
    """Fit every candidate on earlier complete races and keep raw outer-race outputs."""
    columns, dnf_cols = columns_for(contract, masked)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["event_id"], []).append(row)
    order = sorted(grouped, key=lambda event: (grouped[event][0]["prediction_timestamp"], event))
    records = []
    for event_id in order:
        test = sorted(grouped[event_id], key=lambda row: row["driver_id"])
        cutoff = test[0]["prediction_timestamp"]
        train, events = training_rows(rows, cutoff)
        if events < MIN_TRAIN_EVENTS:
            continue
        positions = np.asarray(
            [np.nan if row["label_position"] is None else row["label_position"] for row in test]
        )
        record = RaceRecord(
            event_id,
            cutoff,
            [row["driver_id"] for row in test],
            next(index for index, row in enumerate(test) if row["label_winner"]),
            np.asarray([float(row["label_podium"]) for row in test]),
            positions,
            [row["label_dnf"] for row in test],
            bool(test[0].get("circuit_seen_before")),
            events,
        )
        for name in candidates:
            model = fit_strength(name, train, columns, seed=seed, device=device, hardware=hardware)
            record.utilities[name] = model.utility(test)
            record.devices[name] = model.device
        record.baselines = baseline_marginals(train, test, columns, seed)
        for name in DNF_CANDIDATES:
            record.dnf[name] = fit_dnf(name, train, dnf_cols, seed).predict(test)
        records.append(record)
        if progress is not None:
            progress(f"{contract} {event_id} ({len(records)} outer races)")
    return records


def _pooled(races: list[list[float]]) -> float:
    values = [value for race in races for value in race]
    return float(np.mean(values)) if values else math.inf


def _brier(probability: np.ndarray[Any, Any], labels: list[bool | None]) -> list[float]:
    return [
        (float(p) - float(label)) ** 2
        for p, label in zip(probability, labels, strict=True)
        if label is not None
    ]


def _log_loss(probability: float, label: float) -> float:
    p = min(max(probability, 1e-15), 1 - 1e-15)
    return -(label * math.log(p) + (1 - label) * math.log(1 - p))


def _metrics(records: list[RaceRecord], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    winner_ll, winner_brier, top1, top3 = [], [], [], []
    podium_ll, podium_brier, mae, rps = [], [], [], []
    win_p, win_y, pod_p, pod_y = [], [], [], []
    winner_tail = podium_tail = 0
    sharp: list[dict[str, float]] = []
    for record, values in zip(records, outputs, strict=True):
        winner = np.asarray(values["winner"], dtype=float)
        count = len(winner)
        truth = np.zeros(count)
        truth[record.winner] = 1
        winner_ll.append(-math.log(min(max(float(winner[record.winner]), 1e-15), 1.0)))
        winner_brier.append(float(((winner - truth) ** 2).sum()))
        ranked = np.lexsort((np.asarray(record.drivers), -winner))
        top1.append(float(ranked[0] == record.winner))
        top3.append(float(record.winner in ranked[:3]))
        winner_tail += int(winner[record.winner] < 0.01)
        podium = np.asarray(values["podium"], dtype=float)
        for p, y in zip(podium, record.podium, strict=True):
            podium_ll.append(_log_loss(float(p), float(y)))
            podium_brier.append((float(p) - float(y)) ** 2)
            podium_tail += int(y == 1 and p < 0.01)
        win_p.extend(winner.tolist())
        win_y.extend(truth.astype(bool).tolist())
        pod_p.extend(podium.tolist())
        pod_y.extend(record.podium.astype(bool).tolist())
        expected = np.asarray(values["expected"], dtype=float)
        known = ~np.isnan(record.positions)
        mae.extend(np.abs(expected[known] - record.positions[known]).tolist())
        if "finish" in values:
            cumulative = np.cumsum(np.asarray(values["finish"]), axis=1)
            for driver in np.flatnonzero(known):
                observed = (np.arange(1, count + 1) >= record.positions[driver]).astype(float)
                rps.append(float(((cumulative[driver] - observed) ** 2)[:-1].mean()))
        sharp.append(sharpness(winner, podium))
    winner_ece, winner_curve = expected_calibration_error(win_p, win_y)
    podium_ece, podium_curve = expected_calibration_error(pod_p, pod_y)
    return {
        "races": len(records),
        "winner": {
            "log_loss": float(np.mean(winner_ll)),
            "brier_score": float(np.mean(winner_brier)),
            "top_1_accuracy": float(np.mean(top1)),
            "top_3_accuracy": float(np.mean(top3)),
            "ece": winner_ece,
            "reliability": winner_curve,
            "winner_below_1_percent": winner_tail,
        },
        "podium": {
            "log_loss": float(np.mean(podium_ll)),
            "brier_score": float(np.mean(podium_brier)),
            "ece": podium_ece,
            "reliability": podium_curve,
            "podium_finisher_below_1_percent": podium_tail,
        },
        "finishing_position": {
            "mean_absolute_error": float(np.mean(mae)),
            "ranked_probability_score": float(np.mean(rps)) if rps else None,
        },
        "sharpness": {key: float(np.mean([item[key] for item in sharp])) for key in sharp[0]},
    }


def _race_losses(record: RaceRecord, values: dict[str, Any]) -> dict[str, float]:
    winner = float(values["winner"][record.winner])
    known = ~np.isnan(record.positions)
    podium = np.asarray(values["podium"], dtype=float)
    return {
        "winner_log_loss": -math.log(min(max(winner, 1e-15), 1.0)),
        "podium_brier": float(((podium - record.podium) ** 2).mean()),
        "position_mae": float(
            np.abs(np.asarray(values["expected"])[known] - record.positions[known]).mean()
        ),
    }


def _paired(
    records: list[RaceRecord],
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    result = {}
    rng = np.random.default_rng(seed)
    for metric in ("winner_log_loss", "podium_brier", "position_mae"):
        deltas = np.asarray(
            [
                _race_losses(record, ours)[metric] - _race_losses(record, theirs)[metric]
                for record, ours, theirs in zip(records, candidate, baseline, strict=True)
            ]
        )
        sampled = rng.choice(deltas, size=(BOOTSTRAP_RESAMPLES, len(deltas)), replace=True)
        lower, upper = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
        result[metric] = {
            "mean_loss_delta": float(deltas.mean()),
            "bootstrap_95_percent_interval": [float(lower), float(upper)],
            "paired_races": len(deltas),
        }
    return result


_TASK_KEYS = {
    "winner": ("log_loss", "brier_score", "top_1_accuracy", "top_3_accuracy"),
    "podium": ("log_loss", "brier_score"),
    "finishing_position": ("mean_absolute_error",),
}
_PRIMARY = {
    "winner": "winner_log_loss",
    "podium": "podium_brier",
    "finishing_position": "position_mae",
}


def _gate(
    metrics: dict[str, Any], baselines: dict[str, Any], paired: dict[str, Any], races: int
) -> dict[str, Any]:
    """Frozen v2 thresholds applied task by task against both baselines."""
    tasks = {}
    for task, keys in _TASK_KEYS.items():
        regressions = []
        for name, reference in baselines.items():
            for key in keys:
                ours, theirs = metrics[task][key], reference[task][key]
                worse = ours < theirs - 1e-12 if key.endswith("accuracy") else ours > theirs + 1e-12
                if worse:
                    regressions.append({"baseline": name, "metric": key, "delta": ours - theirs})
        improving = all(
            paired[name][_PRIMARY[task]]["bootstrap_95_percent_interval"][1] < 0
            for name in baselines
        )
        if races < SELECTION_PAIRED_EVENTS:
            status, reason = "no_selection", "insufficient paired Gold races"
        elif regressions:
            status, reason = "no_selection", "task baseline regression"
        elif not improving:
            status, reason = "no_selection", "paired improvement uncertain"
        else:
            status, reason = "provisional", "passes frozen paired thresholds"
        tasks[task] = {"status": status, "reason": reason, "regressions": regressions}
    return tasks


@dataclass
class Calibration:
    temperature: float
    shrink: float
    events: int


def _fit(table: np.ndarray[Any, Any], indices: list[int]) -> Calibration:
    """Minimise mean objective over a (race, temperature, shrink) loss table."""
    if len(indices) < MIN_CALIBRATION_EVENTS:
        return Calibration(1.0, 0.0, len(indices))
    mean = table[indices].mean(axis=0)
    tau, lam = np.unravel_index(int(np.argmin(mean)), mean.shape)
    return Calibration(TEMPERATURES[tau], SHRINKAGE[lam], len(indices))


def analyse(
    records: list[RaceRecord], *, seed: int = 42, candidates: tuple[str, ...] = JOINT_CANDIDATES
) -> dict[str, Any]:
    """Prequential calibration, selection, metrics, gates and live configuration."""
    if len(records) < MIN_SELECTION_EVENTS:
        raise ValueError("too few outer races for cutoff-specific evaluation")
    dnf_used: list[str] = []
    dnf_scores: dict[str, list[list[float]]] = {name: [] for name in DNF_CANDIDATES}
    for record in records:
        history = len(dnf_used)
        if history >= MIN_SELECTION_EVENTS:
            choice = min(
                DNF_CANDIDATES,
                key=lambda name: (_pooled(dnf_scores[name]), name),
            )
        else:
            choice = "prior"
        dnf_used.append(choice)
        for name in DNF_CANDIDATES:
            dnf_scores[name].append(_brier(record.dnf[name], record.dnf_labels))
    priors = []
    model: dict[str, list[list[dict[str, Any]]]] = {name: [] for name in candidates}
    loss: dict[str, np.ndarray[Any, Any]] = {}
    for index, record in enumerate(records):
        dnf = record.dnf[dnf_used[index]]
        count = len(record.drivers)
        rng = np.random.default_rng((seed, index, 0))
        order, retired = sample_orders(
            np.zeros(count), dnf, temperature=1.0, draws=CALIBRATION_DRAWS, rng=rng
        )
        priors.append(marginals(order, retired))
        for name in candidates:
            per_tau = []
            for tau in TEMPERATURES:
                rng = np.random.default_rng((seed, index, 1))
                order, retired = sample_orders(
                    record.utilities[name], dnf, temperature=tau, draws=CALIBRATION_DRAWS, rng=rng
                )
                per_tau.append(marginals(order, retired))
            model[name].append(per_tau)
    for name in candidates:
        table = np.zeros((len(records), len(TEMPERATURES), len(SHRINKAGE)))
        for index, record in enumerate(records):
            for tau in range(len(TEMPERATURES)):
                for lam, shrink in enumerate(SHRINKAGE):
                    values = mix(model[name][index][tau], priors[index], shrink)
                    table[index, tau, lam] = objective(values, record.winner, record.podium)
        loss[name] = table
    unseen = [index for index, record in enumerate(records) if not record.circuit_seen]
    calibrated: dict[str, list[dict[str, Any]]] = {}
    raw: dict[str, list[dict[str, Any]]] = {}
    unseen_effect: dict[str, Any] = {}
    parameters: dict[str, list[dict[str, Any]]] = {}
    tau_one = TEMPERATURES.index(1.0)
    for name in candidates:
        calibrated[name], raw[name], parameters[name] = [], [], []
        rule_scores: list[tuple[float, float]] = []
        for index, record in enumerate(records):
            shared = _fit(loss[name], list(range(index)))
            chosen = shared
            used_rule = False
            if not record.circuit_seen:
                earlier = [item for item in unseen if item < index]
                if len(earlier) >= MIN_UNSEEN_CALIBRATION_EVENTS:
                    rule = _fit(loss[name], earlier)
                    if len(rule_scores) >= 3 and np.mean(
                        [score[1] for score in rule_scores]
                    ) < np.mean([score[0] for score in rule_scores]):
                        chosen, used_rule = rule, True
                    tau_s, lam_s = (
                        TEMPERATURES.index(shared.temperature),
                        SHRINKAGE.index(shared.shrink),
                    )
                    tau_r, lam_r = (
                        TEMPERATURES.index(rule.temperature),
                        SHRINKAGE.index(rule.shrink),
                    )
                    rule_scores.append(
                        (
                            float(loss[name][index, tau_s, lam_s]),
                            float(loss[name][index, tau_r, lam_r]),
                        )
                    )
            tau = TEMPERATURES.index(chosen.temperature)
            calibrated[name].append(mix(model[name][index][tau], priors[index], chosen.shrink))
            raw[name].append(model[name][index][tau_one])
            parameters[name].append(
                {
                    "event_id": record.event_id,
                    "temperature": chosen.temperature,
                    "shrink": chosen.shrink,
                    "calibration_events": chosen.events,
                    "unseen_rule": used_rule,
                }
            )
        unseen_effect[name] = {
            "unseen_races_with_rule_estimate": len(rule_scores),
            "shared_objective": float(np.mean([s[0] for s in rule_scores]))
            if rule_scores
            else None,
            "unseen_rule_objective": float(np.mean([s[1] for s in rule_scores]))
            if rule_scores
            else None,
        }
    for ensemble, members in ENSEMBLE_MEMBERS.items():
        active = [name for name in members if name in candidates]
        if len(active) > 1:
            calibrated[ensemble] = [
                average([calibrated[name][index] for name in active])
                for index in range(len(records))
            ]
    models = list(calibrated)
    race_objective = {
        name: [
            objective(values, record.winner, record.podium)
            for values, record in zip(calibrated[name], records, strict=True)
        ]
        for name in models
    }
    selected, choices = [], []
    for index in range(len(records)):
        if index >= MIN_SELECTION_EVENTS + MIN_CALIBRATION_EVENTS:
            start = MIN_CALIBRATION_EVENTS
            choice = min(models, key=lambda n: (np.mean(race_objective[n][start:index]), n))
        else:
            choice = DEFAULT_CANDIDATE if DEFAULT_CANDIDATE in calibrated else models[0]
        choices.append(choice)
        selected.append(calibrated[choice][index])
    baseline_outputs = {
        name: [record.baselines[name] for record in records] for name in ("heuristic", "logistic")
    }
    baseline_metrics = {
        name: _metrics(records, outputs) for name, outputs in baseline_outputs.items()
    }
    results: dict[str, Any] = {}
    for name, outputs in [*calibrated.items(), ("selected_pipeline", selected)]:
        metrics = _metrics(records, outputs)
        paired = {
            base: _paired(records, outputs, baseline_outputs[base], seed)
            for base in baseline_outputs
        }
        results[name] = {
            "metrics": metrics,
            "paired_uncertainty": paired,
            "formal_gate": _gate(metrics, baseline_metrics, paired, len(records)),
        }
    for name in candidates:
        results[name]["uncalibrated_metrics"] = _metrics(records, raw[name])
        results[name]["unseen_circuit_rule"] = unseen_effect[name]
    start = MIN_CALIBRATION_EVENTS
    primary = min(models, key=lambda n: (np.mean(race_objective[n][start:]), n))
    every = list(range(len(records)))
    final = {name: _fit(loss[name], every) for name in candidates}
    final_unseen = {}
    for name in candidates:
        effect = unseen_effect[name]
        adopt = (
            len(unseen) >= MIN_UNSEEN_CALIBRATION_EVENTS
            and effect["unseen_rule_objective"] is not None
            and effect["unseen_rule_objective"] < effect["shared_objective"]
        )
        final_unseen[name] = _fit(loss[name], unseen).__dict__ if adopt else None
    dnf_final = min(
        DNF_CANDIDATES,
        key=lambda name: (_pooled(dnf_scores[name]), name),
    )
    dnf_metrics = {}
    for name in DNF_CANDIDATES:
        pairs = [
            (float(p), float(y))
            for record in records
            for p, y in zip(record.dnf[name], record.dnf_labels, strict=True)
            if y is not None
        ]
        ece, curve = expected_calibration_error([p for p, _ in pairs], [bool(y) for _, y in pairs])
        dnf_metrics[name] = {
            "labels": len(pairs),
            "brier_score": float(np.mean([(p - y) ** 2 for p, y in pairs])),
            "log_loss": float(np.mean([_log_loss(p, y) for p, y in pairs])),
            "ece": ece,
            "reliability": curve,
            "mean_within_race_standard_deviation": float(
                np.mean([np.std(record.dnf[name]) for record in records])
            ),
        }
    reference = [
        sharpness(np.asarray(values["winner"]), np.asarray(values["podium"]))
        for values in calibrated[primary]
    ]
    strength_model = (
        primary
        if primary in candidates
        else min(candidates, key=lambda n: (np.mean(race_objective[n][start:]), n))
    )
    persistence = persistent_variance(
        records,
        strength_model,
        parameters[strength_model],
        calibrated[strength_model],
        [records[index].dnf[dnf_used[index]] for index in range(len(records))],
        seed=seed,
    )
    return {
        "protocol": PROTOCOL_V3,
        "protocol_sha256": PROTOCOL_V3_SHA256,
        "outer_races": len(records),
        "outer_event_ids": [record.event_id for record in records],
        "unseen_circuit_races": len(unseen),
        "baselines": baseline_metrics,
        "models": results,
        "prequential_choices": choices,
        "calibration_parameters": parameters,
        "dnf": {
            "candidates": dnf_metrics,
            "prequential_choices": dnf_used,
            "live_choice": dnf_final,
        },
        "live": {
            "primary": primary,
            "primary_members": [
                name for name in ENSEMBLE_MEMBERS.get(primary, (primary,)) if name in candidates
            ],
            "strength_model_for_uncertainty": strength_model,
            "calibration": {name: value.__dict__ for name, value in final.items()},
            "unseen_calibration": final_unseen,
            "dnf_model": dnf_final,
            "mean_objective": {
                name: float(np.mean(race_objective[name][start:])) for name in models
            },
            "sharpness_reference": reference,
            "persistent_strength": persistence,
        },
    }


def _residuals(
    records: list[RaceRecord], distributions: list[dict[str, Any]], positions: list[Any]
) -> list[tuple[str, float]]:
    """Standardized residuals of known finishing positions, keyed by driver-season."""
    result = []
    for index in range(MIN_CALIBRATION_EVENTS, len(records)):
        record = records[index]
        finish = np.asarray(distributions[index]["finish"])
        places = np.arange(1, finish.shape[1] + 1)
        mean = finish @ places
        sd = np.sqrt(np.maximum(finish @ places**2 - mean**2, 1e-9))
        observed = positions[index]
        for slot, driver in enumerate(record.drivers):
            if not np.isnan(observed[slot]):
                result.append(
                    (f"{record.event_id[7:11]}:{driver}", (observed[slot] - mean[slot]) / sd[slot])
                )
    return result


def _icc(residuals: list[tuple[str, float]]) -> float:
    """One-way random-effects intraclass correlation over driver-season groups."""
    groups: dict[str, list[float]] = {}
    for key, value in residuals:
        groups.setdefault(key, []).append(value)
    usable = [np.asarray(values) for values in groups.values() if len(values) >= 3]
    if len(usable) < 2:
        return 0.0
    sizes = np.asarray([len(values) for values in usable])
    grand = np.concatenate(usable).mean()
    between = sum(len(v) * (v.mean() - grand) ** 2 for v in usable) / (len(usable) - 1)
    within = sum(((v - v.mean()) ** 2).sum() for v in usable) / (sizes.sum() - len(usable))
    size = (sizes.sum() - (sizes**2).sum() / sizes.sum()) / (len(usable) - 1)
    return float((between - within) / (between + (size - 1) * within))


def persistent_variance(
    records: list[RaceRecord],
    name: str,
    parameters: list[dict[str, Any]],
    distributions: list[dict[str, Any]],
    dnf: list[np.ndarray[Any, Any]],
    *,
    seed: int,
    replicates: int = 20,
) -> dict[str, Any]:
    """Moment-matched persistent driver-season strength variation.

    Observed: intraclass correlation of standardized out-of-fold finishing
    residuals within driver-seasons (classified positions only, so retirement luck
    does not enter). Simulated: the same statistic on histories drawn from the
    calibrated model plus driver-season offsets with standard deviation sigma.
    Sigma is the grid value whose simulated correlation is closest to the observed.
    """
    observed_positions = [record.positions for record in records]
    observed = _icc(_residuals(records, distributions, observed_positions))
    keys = sorted({key for key, _ in _residuals(records, distributions, observed_positions)})
    rng = np.random.default_rng((seed, 3))
    groups: dict[str, list[tuple[str, float]]] = {}
    for item in _residuals(records, distributions, observed_positions):
        groups.setdefault(item[0], []).append(item)
    boot = []
    for _ in range(200):
        sample = rng.choice(len(keys), size=len(keys), replace=True)
        resampled = [
            (f"{draw}", value)
            for draw, index in enumerate(sample)
            for _, value in groups[keys[index]]
        ]
        boot.append(_icc(resampled))
    lower, upper = (float(value) for value in np.quantile(boot, [0.025, 0.975]))
    simulated = []
    for sigma in PERSISTENT_SIGMAS:
        values = []
        for replicate in range(replicates):
            local = np.random.default_rng((seed, 4, replicate))
            offsets: dict[str, float] = {}
            positions = []
            for index, record in enumerate(records):
                season = record.event_id[7:11]
                delta = np.asarray(
                    [
                        offsets.setdefault(f"{season}:{driver}", float(local.normal(0.0, sigma)))
                        for driver in record.drivers
                    ]
                )
                order, retired = sample_orders(
                    record.utilities[name],
                    dnf[index],
                    temperature=parameters[index]["temperature"],
                    draws=1,
                    rng=local,
                    shrink=parameters[index]["shrink"],
                    offsets=delta[None, :],
                )
                place = np.empty(len(record.drivers))
                place[order[0]] = np.arange(1, len(record.drivers) + 1)
                place[retired[0]] = np.nan
                positions.append(place)
            values.append(_icc(_residuals(records, distributions, positions)))
        simulated.append({"sigma": sigma, "icc_mean": float(np.mean(values))})

    def matched(target: float) -> float:
        return min(simulated, key=lambda item: (abs(item["icc_mean"] - target), item["sigma"]))[
            "sigma"
        ]

    best = matched(observed)
    return {
        "method": "moment-matched intraclass correlation of standardized out-of-fold "
        "finishing residuals within driver-seasons",
        "strength_model": name,
        "observed_icc": observed,
        "observed_icc_95_percent_interval": [lower, upper],
        "simulated_icc_by_sigma": simulated,
        "persistent_standard_deviation": best,
        "sigma_interval": sorted([matched(lower), matched(upper)]),
        "grid_edge": best == PERSISTENT_SIGMAS[-1],
    }


def simulation_temperature(
    records: list[RaceRecord],
    name: str,
    dnf_choice: list[str],
    shrink: float,
    sigma: float,
    *,
    seed: int,
) -> dict[str, Any]:
    """Refit temperature with persistent offsets so single-race marginals stay calibrated."""
    if sigma <= 0:
        return {"temperature": None, "losses": []}
    losses = []
    for tau in TEMPERATURES:
        values = []
        for index, record in enumerate(records):
            count = len(record.drivers)
            rng = np.random.default_rng((seed, index, 2))
            offsets = rng.normal(0.0, sigma, size=(CALIBRATION_DRAWS, count))
            dnf = record.dnf[dnf_choice[index]]
            order, retired = sample_orders(
                record.utilities[name],
                dnf,
                temperature=tau,
                draws=CALIBRATION_DRAWS,
                rng=rng,
                shrink=shrink,
                offsets=offsets,
            )
            values.append(objective(marginals(order, retired), record.winner, record.podium))
        losses.append((float(np.mean(values[MIN_CALIBRATION_EVENTS:])), tau))
    return {"temperature": min(losses)[1], "losses": losses}
