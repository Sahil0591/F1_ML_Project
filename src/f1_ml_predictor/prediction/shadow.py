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
from f1_ml_predictor.prediction.joint import marginals, mix, objective
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


def fit_two_stage(records: list[Any], members: list[str], seed: int = 42) -> dict[str, Any]:
    """Two-stage temperatures and shrinkage minimising the v3 objective out of fold.

    Each member is first scaled by its own prequential calibration temperature,
    as the official ensemble is, and the DNF model follows the v3 prequential
    choice for that race.
    """
    report = analyse(records, seed=seed)
    parameters = report["calibration_parameters"]
    used: list[str] = []
    scores: dict[str, list[list[float]]] = {name: [] for name in DNF_CANDIDATES}
    for record in records:
        if len(used) >= MIN_SELECTION_EVENTS:
            used.append(min(DNF_CANDIDATES, key=lambda name: (_pooled(scores[name]), name)))
        else:
            used.append("prior")
        for name in DNF_CANDIDATES:
            scores[name].append(_brier(record.dnf[name], record.dnf_labels))
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
            f"{values['forecast_race_precipitation_mm']:.1f} mm rain over the first two race "
            f"hours, {values['forecast_temperature_2m']:.1f} C, "
            f"wind {values['forecast_wind_speed_10m']:.1f} m/s at the start."
        )
    else:
        lines.append(f"Forecast: {forecast['status']} {forecast.get('reason', '')}".rstrip())
    variants = payload["variants"]
    names = [name for name in ("official", "two_stage", "weather") if "drivers" in variants[name]]
    for name in ("two_stage", "weather"):
        if name not in names:
            lines.append(f"{name}: {variants[name].get('reason', 'skipped')}")
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
