"""Shadow race predictions frozen next to an official post-qualifying run.

Shadows never replace the official numbers. They exist so candidate methods
that did not pass the Gold gate are scored on races they have not seen:

- ``weather``: the weather addendum contract, with the ECMWF run released
  (by the documented margin) before the cutoff, exactly as the historical audit
  chooses it.
- ``two_stage``: the official members pooled into one utility, with one
  temperature for the winner and another for the rest of the order. Both
  temperatures and the shrinkage are fitted on the contract's out-of-fold
  history.
- ``wet_skill``: the pooled utility plus, when the forecast is wet, each
  driver's wet-minus-dry over-performance against the official out-of-fold
  expectation, shrunk towards zero and scaled by a weight fitted on earlier
  forecast-wet races. On a dry forecast it is the pooled model unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np

from f1_ml_predictor.prediction.contracts import AuditedHistory, build_rows, with_weather
from f1_ml_predictor.prediction.evaluation import _brier, _pooled, analyse
from f1_ml_predictor.prediction.joint import marginals, mix, objective, sample_orders
from f1_ml_predictor.prediction.protocol import (
    DNF_CANDIDATES,
    MIN_CALIBRATION_EVENTS,
    MIN_SELECTION_EVENTS,
    SHRINKAGE,
    WEATHER_CONTRACT,
)
from f1_ml_predictor.prediction.workspace import out_of_fold_path
from f1_ml_predictor.trust.historical import _immutable, _json
from f1_ml_predictor.trust.weather_history import (
    ENDPOINT,
    MODEL,
    RELEASE_MARGIN,
    _parameters,
    _retain,
    forecast_run,
    race_weather,
)

SHADOW_VERSION = "shadow-v1"
WINNER_TEMPERATURES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15)
REST_TEMPERATURES = (0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5)
TWO_STAGE_SHRINKAGE = SHRINKAGE[:3]
FIT_DRAWS = 2048
POOLED_TEMPERATURES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5)
WET_THRESHOLD_MM = 0.5
WET_SHRINK_RACES = 5.0
WET_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
MIN_WET_RACES = 4
_OUTPUT = Path("data/predictions/development/shadows")


def live_weather(
    root: Path,
    latitude: float,
    longitude: float,
    cutoff: datetime,
    race_start: datetime,
    *,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """The archived run the historical audit would choose for this cutoff, retained."""
    run = forecast_run(cutoff)
    record: dict[str, Any] = {
        "model": MODEL,
        "run_initialized_at": run.isoformat(),
        "available_at": run + RELEASE_MARGIN,
        "latitude": latitude,
        "longitude": longitude,
    }
    owned = http_client is None
    client = http_client or httpx.Client(timeout=60.0)
    try:
        response = client.get(ENDPOINT, params=_parameters(latitude, longitude, run))
        document = _retain(root, response)
        values = race_weather(json.loads(response.content), race_start)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return {**record, "status": "unavailable", "reason": str(exc)}
    finally:
        if owned:
            client.close()
    complete = all(value is not None for value in values.values())
    return {
        **record,
        "status": "audited" if complete else "incomplete",
        "document": document,
        "values": values,
    }


def two_stage_orders(
    utility: np.ndarray[Any, Any],
    dnf: np.ndarray[Any, Any],
    winner_temperature: float,
    rest_temperature: float,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Plackett-Luce orders whose first pick uses its own temperature.

    Retired drivers are placed last, as in the official simulation; the winner
    is drawn among the classified drivers of each draw.
    """
    count = len(utility)
    retired = rng.random((draws, count)) < dnf
    first = np.where(
        retired, -np.inf, utility / winner_temperature + rng.gumbel(size=retired.shape)
    )
    winner = np.argmax(first, axis=1)
    key = np.where(retired, -1e9, 0.0) + utility / rest_temperature + rng.gumbel(size=retired.shape)
    rows = np.arange(draws)
    nobody = retired.all(axis=1)
    key[rows, winner] = np.where(nobody, key[rows, winner], np.inf)
    return np.argsort(-key, axis=1, kind="stable"), retired


def _pooled_utility(utilities: list[np.ndarray[Any, Any]], temperatures: list[float]) -> Any:
    return np.mean([u / t for u, t in zip(utilities, temperatures, strict=True)], axis=0)


def _two_stage_values(
    utility: np.ndarray[Any, Any],
    dnf: np.ndarray[Any, Any],
    parameters: dict[str, float],
    *,
    draws: int,
    seed: Any,
) -> dict[str, Any]:
    order, retired = two_stage_orders(
        utility,
        dnf,
        parameters["winner_temperature"],
        parameters["rest_temperature"],
        draws=draws,
        rng=np.random.default_rng(seed),
    )
    values = marginals(order, retired)
    if parameters["shrink"] > 0:
        prior_order, prior_retired = two_stage_orders(
            np.zeros_like(utility), dnf, 1.0, 1.0, draws=draws, rng=np.random.default_rng(seed)
        )
        values = mix(values, marginals(prior_order, prior_retired), parameters["shrink"])
    return values


def _dnf_choices(records: list[Any]) -> list[str]:
    """The v3 prequential DNF model choice for each out-of-fold race."""
    used: list[str] = []
    scores: dict[str, list[list[float]]] = {name: [] for name in DNF_CANDIDATES}
    for record in records:
        if len(used) >= MIN_SELECTION_EVENTS:
            used.append(min(DNF_CANDIDATES, key=lambda name: (_pooled(scores[name]), name)))
        else:
            used.append("prior")
        for name in DNF_CANDIDATES:
            scores[name].append(_brier(record.dnf[name], record.dnf_labels))
    return used


def fit_two_stage(records: list[Any], members: list[str], seed: int = 42) -> dict[str, Any]:
    """Two-stage temperatures and shrinkage minimising the v3 objective out of fold.

    Each member is first scaled by its own prequential calibration temperature,
    as the official ensemble is, and the DNF model follows the v3 prequential
    choice for that race.
    """
    report = analyse(records, seed=seed)
    parameters = report["calibration_parameters"]
    used = _dnf_choices(records)
    grid = [
        (first, rest, shrink)
        for first in WINNER_TEMPERATURES
        for rest in REST_TEMPERATURES
        for shrink in TWO_STAGE_SHRINKAGE
    ]
    totals = dict.fromkeys(grid, 0.0)
    races = range(MIN_CALIBRATION_EVENTS, len(records))
    for index in races:
        record = records[index]
        utility = _pooled_utility(
            [record.utilities[name] for name in members],
            [parameters[name][index]["temperature"] for name in members],
        )
        dnf = record.dnf[used[index]]
        for first, rest, shrink in grid:
            values = _two_stage_values(
                utility,
                dnf,
                {"winner_temperature": first, "rest_temperature": rest, "shrink": shrink},
                draws=FIT_DRAWS,
                seed=(seed, index, 9),
            )
            totals[(first, rest, shrink)] += objective(values, record.winner, record.podium)
    first, rest, shrink = min(grid, key=lambda item: (totals[item], item))
    return {
        "winner_temperature": first,
        "rest_temperature": rest,
        "shrink": shrink,
        "mean_objective": totals[(first, rest, shrink)] / len(races),
        "races": len(races),
        "members": list(members),
    }


def two_stage_parameters(
    root: Path, history: AuditedHistory, model: Any, seed: int = 42
) -> dict[str, Any]:
    """Cached two-stage fit for the official model's own out-of-fold records."""
    path = out_of_fold_path(
        root,
        history,
        model.contract,
        model.dataset_sha256,
        masked=model.masked,
        seed=seed,
        candidates=model.candidates,
    )
    if not path.exists():
        raise ValueError("the official contract has no cached out-of-fold records")
    key = hashlib.sha256(
        _json({"oof": path.name, "members": model.members, "version": SHADOW_VERSION})
    ).hexdigest()
    cache = path.parent / f"two-stage-{key}.json"
    if cache.exists():
        return dict(json.loads(cache.read_text(encoding="utf-8")))
    fitted = fit_two_stage(joblib.load(path), list(model.members), seed)
    _immutable(cache, _json(fitted))
    return fitted


def wet_forecast(record: dict[str, Any] | None) -> bool | None:
    """Rain before plus during the race at or above the threshold; None if unknown."""
    if record is None or record.get("status") != "audited":
        return None
    values = record["values"]
    rain = values["forecast_pre_race_precipitation_mm"] + values["forecast_race_precipitation_mm"]
    return bool(rain >= WET_THRESHOLD_MM)


def wet_skills(
    records: list[Any], expected: list[Any], wet: list[bool | None], upto: int
) -> dict[str, float]:
    """Shrunk wet-minus-dry finishing gain over the expectation, from races before ``upto``."""
    gains: dict[str, dict[bool, list[float]]] = {}
    for index in range(upto):
        if wet[index] is None:
            continue
        record = records[index]
        for position, driver in enumerate(record.drivers):
            if np.isnan(record.positions[position]):
                continue
            gain = float(expected[index]["expected"][position] - record.positions[position])
            gains.setdefault(driver, {True: [], False: []})[bool(wet[index])].append(gain)
    result = {}
    for driver, split in gains.items():
        if split[True] and split[False]:
            count = len(split[True])
            weight = count / (count + WET_SHRINK_RACES)
            result[driver] = float((np.mean(split[True]) - np.mean(split[False])) * weight)
    return result


def _pooled_values(
    utility: np.ndarray[Any, Any],
    dnf: np.ndarray[Any, Any],
    temperature: float,
    shrink: float,
    *,
    draws: int,
    seed: Any,
) -> dict[str, Any]:
    order, retired = sample_orders(
        utility, dnf, temperature=temperature, draws=draws, rng=np.random.default_rng(seed)
    )
    values = marginals(order, retired)
    if shrink > 0:
        prior = sample_orders(
            np.zeros_like(utility),
            dnf,
            temperature=1.0,
            draws=draws,
            rng=np.random.default_rng(seed),
        )
        values = mix(values, marginals(*prior), shrink)
    return values


def _driver_value(table: dict[str, float], driver: str) -> float:
    """Gold IDs such as lindblad also match capture IDs such as arvid_lindblad."""
    if driver in table:
        return table[driver]
    matches = [key for key in table if driver.endswith(f"_{key}")]
    return table[matches[0]] if len(matches) == 1 else 0.0


def fit_wet_skill(
    records: list[Any],
    members: list[str],
    primary: str,
    weather: dict[str, dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    """Pooled temperature, shrinkage and wet-skill weight, fitted out of fold.

    Skills for each wet race use only earlier races. The weight minimises the v3
    objective over the forecast-wet races with at least MIN_WET_RACES labelled
    wet races before them.
    """
    outputs: dict[str, list[dict[str, Any]]] = {}
    report = analyse(records, seed=seed, outputs=outputs)
    parameters = report["calibration_parameters"]
    expected = outputs[primary]
    used = _dnf_choices(records)
    wet = [wet_forecast(weather.get(record.event_id)) for record in records]
    utilities = [
        _pooled_utility(
            [record.utilities[name] for name in members],
            [parameters[name][index]["temperature"] for name in members],
        )
        for index, record in enumerate(records)
    ]
    grid = [(tau, shrink) for tau in POOLED_TEMPERATURES for shrink in SHRINKAGE[:4]]
    totals = dict.fromkeys(grid, 0.0)
    races = range(MIN_CALIBRATION_EVENTS, len(records))
    for index in races:
        dnf = records[index].dnf[used[index]]
        for tau, shrink in grid:
            values = _pooled_values(
                utilities[index], dnf, tau, shrink, draws=FIT_DRAWS, seed=(seed, index, 7)
            )
            totals[(tau, shrink)] += objective(values, records[index].winner, records[index].podium)
    temperature, shrink = min(grid, key=lambda item: (totals[item], item))
    scored = [
        index
        for index in races
        if wet[index] and sum(1 for flag in wet[:index] if flag) >= MIN_WET_RACES
    ]
    by_weight = dict.fromkeys(WET_WEIGHTS, 0.0)
    for index in scored:
        skills = wet_skills(records, expected, wet, index)
        bonus = np.asarray([skills.get(driver, 0.0) for driver in records[index].drivers])
        dnf = records[index].dnf[used[index]]
        for weight in WET_WEIGHTS:
            values = _pooled_values(
                utilities[index] + weight * bonus,
                dnf,
                temperature,
                shrink,
                draws=FIT_DRAWS,
                seed=(seed, index, 7),
            )
            by_weight[weight] += objective(values, records[index].winner, records[index].podium)
    weight = min(WET_WEIGHTS, key=lambda item: (by_weight[item], item)) if scored else 0.0
    return {
        "temperature": temperature,
        "shrink": shrink,
        "weight": weight,
        "wet_races_scored": len(scored),
        "mean_objective_by_weight": (
            {str(key): value / len(scored) for key, value in by_weight.items()} if scored else {}
        ),
        "skills": wet_skills(records, expected, wet, len(records)),
        "members": list(members),
        "primary": primary,
    }


def wet_skill_parameters(
    root: Path, history: AuditedHistory, model: Any, seed: int = 42
) -> dict[str, Any] | None:
    """Cached wet-skill fit for the official model's out-of-fold records, or None."""
    if history.weather_audit is None:
        return None
    path = out_of_fold_path(
        root,
        history,
        model.contract,
        model.dataset_sha256,
        masked=model.masked,
        seed=seed,
        candidates=model.candidates,
    )
    if not path.exists():
        raise ValueError("the official contract has no cached out-of-fold records")
    key = hashlib.sha256(
        _json(
            {
                "oof": path.name,
                "members": model.members,
                "primary": model.primary,
                "weather_audit": history.weather_audit,
                "version": SHADOW_VERSION,
            }
        )
    ).hexdigest()
    cache = path.parent / f"wet-skill-{key}.json"
    if cache.exists():
        return dict(json.loads(cache.read_text(encoding="utf-8")))
    fitted = fit_wet_skill(
        joblib.load(path), list(model.members), model.primary, history.weather, seed
    )
    _immutable(cache, _json(fitted))
    return fitted


def _summary(rows: list[dict[str, Any]], values: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        row["driver_id"]: {
            "winner": float(values["winner"][index]),
            "podium": float(values["podium"][index]),
            "expected": float(values["expected"][index]),
            "dnf": float(values["dnf"][index]),
        }
        for index, row in enumerate(rows)
    }


def publish_shadows(
    root: Path,
    *,
    history: AuditedHistory,
    datasets: dict[str, dict[str, Any]],
    model: Any,
    model_factory: Callable[[str, list[dict[str, Any]]], Any],
    official: dict[str, Any],
    rows: list[dict[str, Any]],
    weekend_values: dict[str, dict[str, Any]],
    event: Any,
    circuit_id: str,
    cutoff: datetime,
    race_start: datetime,
    location: tuple[float, float],
    run_id: str,
    draws: int,
    seed: int,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Freeze the shadow variants for one official race run."""
    unseen = not bool(rows[0]["circuit_seen_before"])
    variants: dict[str, Any] = {"official": {"drivers": _summary(rows, official)}}
    components = model.components(rows, unseen)
    parameters = two_stage_parameters(root, history, model, seed)
    utility = _pooled_utility([item[0] for item in components], [item[1] for item in components])
    variants["two_stage"] = {
        "parameters": parameters,
        "drivers": _summary(
            rows,
            _two_stage_values(utility, model.dnf.predict(rows), parameters, draws=draws, seed=seed),
        ),
    }
    weather = live_weather(root, *location, cutoff, race_start, http_client=http_client)
    wet_fit = wet_skill_parameters(root, history, model, seed)
    if wet_fit is None:
        variants["wet_skill"] = {"status": "skipped", "reason": "no weather audit"}
    else:
        applied = bool(wet_forecast(weather))
        bonus = np.asarray([_driver_value(wet_fit["skills"], row["driver_id"]) for row in rows])
        variants["wet_skill"] = {
            "status": "published",
            "applied": applied,
            "parameters": {key: value for key, value in wet_fit.items() if key != "skills"},
            "skills": {
                row["driver_id"]: float(value) for row, value in zip(rows, bonus, strict=True)
            },
            "drivers": _summary(
                rows,
                _pooled_values(
                    utility + wet_fit["weight"] * bonus if applied else utility,
                    model.dnf.predict(rows),
                    wet_fit["temperature"],
                    wet_fit["shrink"],
                    draws=draws,
                    seed=seed,
                ),
            ),
        }
    if weather["status"] != "audited":
        variants["weather"] = {"status": "skipped", "reason": "no complete forecast"}
    elif WEATHER_CONTRACT not in datasets:
        variants["weather"] = {
            "status": "skipped",
            "reason": "no weather audit for the current Gold races; run "
            "scripts/audit-historical-weather.py",
        }
    else:
        weather_rows, _ = build_rows(
            history,
            WEATHER_CONTRACT,
            event=event,
            circuit_id=circuit_id,
            cutoff=cutoff,
            roster={row["driver_id"]: row["constructor_id"] for row in rows},
            weekend=with_weather(weekend_values, weather),
        )
        weather_model = model_factory(WEATHER_CONTRACT, weather_rows)
        values = weather_model.distribution(
            weather_rows, weather_model.components(weather_rows, unseen), draws=draws, seed=seed
        )
        variants["weather"] = {
            "status": "published",
            "dataset_sha256": weather_model.dataset_sha256,
            "drivers": _summary(weather_rows, values),
        }
    payload = {
        "version": SHADOW_VERSION,
        "official_run_id": run_id,
        "event_id": event.partition(),
        "cutoff_kind": model.contract,
        "prediction_timestamp_utc": cutoff.isoformat(),
        "race_start_utc": race_start.isoformat(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "forecast": {
            **{key: value for key, value in weather.items() if key != "available_at"},
            "available_at": weather["available_at"].isoformat(),
        },
        "variants": variants,
    }
    path = root / _OUTPUT / event.partition() / f"{run_id}.json"
    _immutable(path, _json(payload))
    return {"path": path.relative_to(root).as_posix(), "forecast": payload["forecast"]}


def render_shadows(root: Path, result: dict[str, Any], top: int = 8) -> str:
    """Plain-text forecast and side-by-side win/podium table for the console."""
    if "path" not in result:
        return f"shadow predictions: {result.get('status')} ({result.get('reason')})"
    payload = json.loads((root / result["path"]).read_text(encoding="utf-8"))
    forecast = payload["forecast"]
    lines = ["", "## Shadow predictions (not official)", ""]
    if forecast["status"] == "audited":
        values = forecast["values"]
        lines.append(
            f"Forecast (ECMWF run {forecast['run_initialized_at']}): "
            f"{values['forecast_pre_race_precipitation_mm']:.1f} mm rain in the six hours "
            f"before the start, {values['forecast_race_precipitation_mm']:.1f} mm over the "
            f"first two race hours, {values['forecast_temperature_2m']:.1f} C, "
            f"wind {values['forecast_wind_speed_10m']:.1f} m/s at the start."
        )
    else:
        lines.append(f"Forecast: {forecast['status']} {forecast.get('reason', '')}".rstrip())
    variants = payload["variants"]
    shadows = ("two_stage", "weather", "wet_skill")
    names = ["official", *(name for name in shadows if "drivers" in variants[name])]
    for name in shadows:
        if name not in names:
            lines.append(f"{name}: {variants[name].get('reason', 'skipped')}")
    if "drivers" in variants["wet_skill"]:
        wet = variants["wet_skill"]
        state = "applied" if wet["applied"] else "not applied (dry forecast)"
        lines.append(
            f"wet_skill: {state}, weight {wet['parameters']['weight']} on each driver's "
            "wet-minus-dry gain in places"
        )
    official = variants["official"]["drivers"]
    order = sorted(official, key=lambda driver: -official[driver]["winner"])[:top]
    lines += [
        "",
        "| Driver | " + " | ".join(f"{name} win | {name} podium" for name in names) + " |",
        "| --- | " + " | ".join("---: | ---:" for _ in names) + " |",
    ]
    for driver in order:
        cells = []
        for name in names:
            item = variants[name]["drivers"][driver]
            cells.append(f"{item['winner']:.1%} | {item['podium']:.1%}")
        lines.append(f"| {driver} | " + " | ".join(cells) + " |")
    lines.append(f"\nshadow file: {result['path']}")
    return "\n".join(lines)
