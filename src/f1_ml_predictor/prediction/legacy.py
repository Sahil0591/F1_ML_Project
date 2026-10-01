"""The c52b674 development pipeline, kept unchanged so its forecasts stay reproducible.

``predict-next-race --methodology c52b674`` reruns it. The current methodology is
in ``prediction.pipeline``.

Generate a development-only prediction for the next race and its season.

The pipeline freezes a point-in-time snapshot, fits task-specific development
models on the latest immutable Gold versions, samples coherent race orders with
the existing joint sampler and runs the existing championship simulator. Every
artifact is labelled ``development_only``; nothing here satisfies a validation
gate or promotes a model.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from f1_ml_predictor.benchmarks.builder import file_sha256
from f1_ml_predictor.models.development import _check_race
from f1_ml_predictor.models.hardware import library_versions
from f1_ml_predictor.models.protocol import PROTOCOL, PROTOCOL_SHA256
from f1_ml_predictor.prediction.history import (
    GoldVersion,
    audited_outcomes,
    choose_dnf_model,
    choose_position_backend,
    latest_dnf_directory,
    latest_gold_directory,
    load_gold_version,
    reference_run,
)
from f1_ml_predictor.prediction.live_features import (
    LIVE_SNAPSHOT_VERSION,
    PRE_QUALIFYING,
    ScheduledEvent,
    apply_mask,
    availability_mask,
    build_live_rows,
    freeze_snapshot,
    load_schedule,
    missing_summary,
)
from f1_ml_predictor.prediction.race import (
    COMPOSED_MODEL_VERSION,
    ComposedRaceModel,
    baseline_predictions,
    evaluate_availability_variant,
    feature_importance,
    fit_composed_model,
)
from f1_ml_predictor.prediction.report import render_report
from f1_ml_predictor.prediction.season import (
    ELIGIBILITY_POLICY,
    points_rules,
    published_standings,
    remaining_sessions,
)
from f1_ml_predictor.scoring.ledger import load_scoring_ledger
from f1_ml_predictor.simulation import EventSimulation, simulate_championship
from f1_ml_predictor.time import require_known_by, require_utc
from f1_ml_predictor.trust.scheduler import scheduler_status

STATUS = "development_only"
WARNING = (
    "DEVELOPMENT ONLY. These outputs come from models that have not passed the frozen "
    "Gold selection gates or prospective confirmation. They are not validated race or "
    "championship forecasts."
)
_OUTPUT_ROOT = Path("data/predictions/development/next_race")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unserializable {type(value).__name__}")


def _write_json(path: Path, value: Any) -> str:
    data = json.dumps(value, sort_keys=True, indent=2, default=_json_default, allow_nan=False)
    with path.open("xb") as handle:
        handle.write(data.encode("utf-8"))
    return file_sha256(path)


def _capture_base(
    root: Path, state: dict[str, Any], target: ScheduledEvent, clock: datetime
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]] | None:
    """Return the latest certified post-qualifying capture for the target race, if any."""
    entry = state.get("events", {}).get(target.event.partition())
    if not entry or entry.get("race_start") != target.race_start.isoformat():
        return None
    captures = [
        capture
        for capture in entry.get("captures", [])
        if capture.get("cutoff_kind") in {"post_qualifying", "pre_race"}
        and capture.get("features")
        and datetime.fromisoformat(capture["captured_at"]) <= clock
    ]
    if not captures:
        return None
    capture = max(captures, key=lambda item: item["captured_at"])
    path = root / capture["features"]["path"]
    if file_sha256(path) != capture["features"]["sha256"]:
        raise ValueError("certified capture features changed since collection")
    rows = pq.read_table(path).to_pylist()
    if {row["event_id"] for row in rows} != {target.event.partition()}:
        raise ValueError("certified capture belongs to another event")
    return capture, {row["driver_id"]: row for row in rows}


def _latest_roster(
    rows: list[dict[str, Any]], season: int, cutoff: datetime
) -> tuple[dict[str, str], str]:
    earlier = [
        row
        for row in rows
        if row["event_id"].startswith(f"season={season}/")
        and row["prediction_timestamp"] < cutoff
        and row["label_available_at"] <= cutoff
    ]
    if not earlier:
        raise ValueError("no audited same-season race exists to supply a roster")
    latest = max(earlier, key=lambda row: row["prediction_timestamp"])["event_id"]
    roster = {
        row["driver_id"]: row["constructor_id"] for row in earlier if row["event_id"] == latest
    }
    return roster, latest


def _race_rows(
    model: ComposedRaceModel,
    rows: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    distribution = model.distribution(rows, draws=draws, seed=seed)
    for row in distribution:
        values = row["finish_distribution"]
        row["most_likely_position"] = int(np.argmax(values)) + 1
        row["position_interval_80"] = [
            int(np.searchsorted(np.cumsum(values), quantile - 1e-12)) + 1 for quantile in (0.1, 0.9)
        ]
    _check_race(distribution)
    return distribution


def _model_record(root: Path, run_dir: Path, name: str, model: ComposedRaceModel) -> dict[str, Any]:
    path = root / "models/development/next_race" / run_dir.name / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "metadata": model.metadata,
    }


def predict_next_race_c52b674(
    root: Path,
    *,
    season: int | None = None,
    simulations: int = 100000,
    seed: int = 42,
    draws: int = 65536,
    championship_orders: int = 8192,
    device: str = "auto",
    gold_dir: Path | None = None,
    dnf_dir: Path | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Freeze the latest snapshot and publish development race and title outputs."""
    root = root.resolve()
    clock = (now or (lambda: datetime.now(UTC)))()
    require_utc(clock, "prediction clock")
    state = scheduler_status(root)
    observation = state.get("schedule_observation")
    if not observation:
        raise ValueError("no retained schedule observation; run collect-next-race first")
    schedule, schedule_source = load_schedule(root, observation)
    observed_at = datetime.fromisoformat(schedule_source["captured_at"])
    require_known_by(observed_at, clock)
    upcoming = [
        item
        for item in schedule
        if item.race_start > clock and (season is None or item.event.season == season)
    ]
    if not upcoming:
        raise ValueError("the retained schedule has no upcoming race")
    target = upcoming[0]

    capture = _capture_base(root, state, target, clock)
    if capture is None:
        cutoff, cutoff_kind, base, capture_record = clock, PRE_QUALIFYING, None, None
    else:
        capture_record, base = capture
        cutoff = next(iter(base.values()))["prediction_timestamp"]
        cutoff_kind = capture_record["cutoff_kind"]
    require_known_by(observed_at, cutoff)

    version = load_gold_version(gold_dir or latest_gold_directory(root))
    dnf_version = load_gold_version(dnf_dir or latest_dnf_directory(root))
    outcomes = audited_outcomes(root, version)
    ledger = load_scoring_ledger(
        root / "data/audit/scoring_rules.json", root / "data/audit/event_points_evidence.json"
    )
    if ledger.sha256 != version.manifest.get("scoring_ledger_sha256"):
        raise ValueError("scoring ledger differs from the Gold version; rebuild Gold scoring first")
    if base is None:
        roster, roster_basis = _latest_roster(version.rows, target.event.season, cutoff)
        roster_source = f"latest_audited_gold_event_roster:{roster_basis}"
    else:
        roster = {driver: row["constructor_id"] for driver, row in base.items()}
        roster_source = "certified_post_qualifying_capture"

    reference_path, reference = reference_run(root, version)
    dnf_reference_path, dnf_reference = reference_run(root, dnf_version)
    position_choice = choose_position_backend(reference, "post_qualifying")
    dnf_choice = choose_dnf_model(dnf_reference, "post_qualifying")
    hardware_path = root / "models/experiments/hardware.json"
    hardware = (
        json.loads(hardware_path.read_text(encoding="utf-8"))
        if device != "cpu" and hardware_path.exists()
        else None
    )

    variants: dict[str, dict[str, Any]] = {}
    plans: list[tuple[str, dict[str, dict[str, Any]] | None]] = [(PRE_QUALIFYING, None)]
    if base is not None:
        plans.insert(0, (cutoff_kind, base))
    for kind, variant_base in plans:
        rows, provenance = build_live_rows(
            version,
            outcomes,
            ledger,
            target=target,
            cutoff=cutoff,
            cutoff_kind=kind,
            roster=roster,
            opened_at=observed_at,
            base=variant_base,
        )
        masked = availability_mask(version, rows)
        snapshot = freeze_snapshot(
            root,
            version,
            rows,
            {
                "event_id": target.event.partition(),
                "race_name": target.race_name,
                "cutoff_kind": kind,
                "prediction_timestamp_utc": cutoff.isoformat(),
                "dataset_version": version.dataset_version,
                "feature_schema_version": version.manifest["scoring_version"],
                "scoring_ledger_sha256": ledger.sha256,
                "audited_registry_sha256": version.manifest["catalog_sha256"],
                "schedule_source": schedule_source,
                "capture": None
                if variant_base is None or capture_record is None
                else {
                    "bundle": capture_record["bundle"],
                    "manifest_sha256": capture_record["manifest_sha256"],
                    "captured_at": capture_record["captured_at"],
                    "features": capture_record["features"],
                },
                "roster_source": roster_source,
                "masked_for_training": list(masked),
                "provenance": provenance,
            },
        )
        dnf_masked = tuple(name for name in masked if name in dnf_version.feature_columns)
        model = fit_composed_model(
            apply_mask(version.rows, masked),
            rows,
            apply_mask(dnf_version.rows, dnf_masked),
            backend=position_choice["backend"],
            dnf_name=dnf_choice["model"],
            cutoff=cutoff,
            cutoff_kind=kind,
            seed=seed,
            device=device,
            hardware=hardware,
        )
        variants[kind] = {
            "rows": rows,
            "provenance": provenance,
            "masked": masked,
            "snapshot": snapshot,
            "model": model,
            "missing": missing_summary(version, rows, provenance, masked),
            "evaluation": evaluate_availability_variant(
                root, version, dnf_version, masked, backend=position_choice["backend"], seed=seed
            ),
            "baselines": baseline_predictions(
                apply_mask(version.rows, masked), rows, cutoff=cutoff, seed=seed
            ),
            "importance": feature_importance(model),
        }

    run_id = uuid.uuid4().hex
    run_dir = root / _OUTPUT_ROOT / target.event.partition() / cutoff_kind / run_id
    run_dir.mkdir(parents=True)
    models = {
        kind: _model_record(root, run_dir, kind, variant["model"])
        for kind, variant in variants.items()
    }
    model_ids = {kind: f"{run_id}:{kind}:{COMPOSED_MODEL_VERSION}" for kind in variants}
    primary = variants[cutoff_kind]
    race = _race_rows(primary["model"], primary["rows"], draws=draws, seed=seed)

    standings, standings_notes, standings_sources = published_standings(
        ledger, target.event, cutoff, roster
    )
    sessions = []
    events = []
    for item, kind, start in remaining_sessions(schedule, target, cutoff):
        variant = cutoff_kind if item.event == target.event and kind == "race" else PRE_QUALIFYING
        rules, rules_status = points_rules(ledger, item.event, kind)
        session_seed = seed + 1000 * item.event.round + (1 if kind == "sprint" else 0)
        orders = variants[variant]["model"].orders(
            variants[variant]["rows"], draws=championship_orders, seed=session_seed
        )
        events.append(
            EventSimulation(
                event_id=item.event,
                scheduled_at=start,
                available_at=cutoff,
                driver_constructors=roster,
                sampled_orders=orders,
                points_eligible_samples=tuple(tuple(roster) for _ in orders),
                rules=rules,
                source_hash=models[variant]["sha256"],
                model_id=model_ids[variant],
                eligibility_policy=ELIGIBILITY_POLICY,
            )
        )
        sessions.append(
            {
                "event_id": item.event.partition(),
                "race_name": item.race_name,
                "session": kind,
                "scheduled_at": start.isoformat(),
                "model_variant": variant,
                "model_id": model_ids[variant],
                "orders": championship_orders,
                "order_seed": session_seed,
                "rules_id": rules.rules_id,
                "rules_status": rules_status,
                "points_by_position": list(rules.points_by_position),
            }
        )
    championship = simulate_championship(
        standings, events, prediction_timestamp=cutoff, simulations=simulations, seed=seed
    )

    created_at = datetime.now(UTC)
    common = {
        "validation_status": STATUS,
        "model_run_id": run_id,
        "model_version": COMPOSED_MODEL_VERSION,
        "dataset_manifest_hash": version.manifest_sha256,
        "feature_schema_version": version.manifest["scoring_version"],
        "snapshot_sha256": primary["snapshot"]["sha256"],
    }
    baselines = primary["baselines"]
    constructors = {row["driver_id"]: row["constructor_id"] for row in primary["rows"]}
    table_rows = [
        {
            "event_id": target.event.partition(),
            "race_name": target.race_name,
            "driver_id": row["driver_id"],
            "constructor_id": constructors[row["driver_id"]],
            "prediction_timestamp": cutoff,
            "cutoff_kind": cutoff_kind,
            "generated_at": created_at,
            "win_probability": row["winner_probability"],
            "podium_probability": row["podium_probability"],
            "dnf_probability": row["dnf_probability"],
            "dnf_model_probability": row["dnf_model_probability"],
            "expected_finish": row["expected_position"],
            "most_likely_position": row["most_likely_position"],
            "finishing_position_distribution": row["finish_distribution"],
            "position_score": row["position_score"],
            "logistic_win_probability": baselines[row["driver_id"]]["logistic_win_probability"],
            "logistic_podium_probability": baselines[row["driver_id"]][
                "logistic_podium_probability"
            ],
            "linear_finish_rank": baselines[row["driver_id"]]["linear_finish_rank"],
            "heuristic_win_probability": baselines[row["driver_id"]]["heuristic_win_probability"],
            "position_backend": position_choice["backend"],
            "dnf_model": dnf_choice["model"],
            **common,
        }
        for row in race
    ]
    predictions_path = run_dir / "predictions.parquet"
    pq.write_table(pa.Table.from_pylist(table_rows), predictions_path)
    race_payload = {
        **common,
        "event_id": target.event.partition(),
        "race_name": target.race_name,
        "race_start": target.race_start.isoformat(),
        "cutoff_kind": cutoff_kind,
        "prediction_timestamp_utc": cutoff.isoformat(),
        "draws": draws,
        "seed": seed,
        "temperature": primary["model"].temperature,
        "simulation_standard_error_max": 0.5 / draws**0.5,
        "distribution_policy": (
            "independent sampled DNF from the task DNF model; Plackett-Luce order from the "
            "position model; sampled retirees trail finishers; total modelled order, not FIA "
            "classification"
        ),
        "drivers": race,
    }
    race_sha = _write_json(run_dir / "race_distribution.json", race_payload)
    championship_payload = {
        "validation_status": STATUS,
        "validated_forecast": False,
        "model_run_id": run_id,
        "model_ids": model_ids,
        "race_distribution_sha256": race_sha,
        "standings_source": {
            "scoring_ledger_sha256": ledger.sha256,
            "rounds": standings_sources,
            "notes": standings_notes,
        },
        "sessions": sessions,
        "eligibility_policy": ELIGIBILITY_POLICY,
        "starting_points": standings.to_dict(),
        "assumptions": [
            "Races after the next one use the pre-qualifying model with form held at the cutoff.",
            "Sprints reuse the race model with the audited sprint points table.",
            "Every remaining round uses the full-distance table of the latest audited "
            "season rule; shortened races, cancellations and penalties are not simulated.",
            "Every entered driver is assumed points eligible at their sampled position.",
            "The current roster is assumed for every remaining round.",
            "Remaining events are independent; season-level shocks are omitted.",
        ],
        "simulator": championship.to_dict(),
    }
    championship_sha = _write_json(run_dir / "championship.json", championship_payload)

    checks = validate_publication(
        root,
        race=race,
        variants=variants,
        cutoff=cutoff,
        observed_at=observed_at,
        version=version,
        dnf_version=dnf_version,
        events=events,
        model_ids=model_ids,
        models=models,
        ledger_sources=standings_sources,
        artifacts=[
            predictions_path,
            run_dir / "race_distribution.json",
            run_dir / "championship.json",
        ],
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    manifest = {
        **common,
        "warning": WARNING,
        "validated_forecast": False,
        "created_at": created_at.isoformat(),
        "git_commit": commit,
        "event": {
            "event_id": target.event.partition(),
            "race_name": target.race_name,
            "circuit_id": target.circuit_id,
            "race_start": target.race_start.isoformat(),
            "qualifying_start": None
            if target.qualifying_start is None
            else target.qualifying_start.isoformat(),
            "qualifying_started_before_cutoff": target.qualifying_start is not None
            and target.qualifying_start <= cutoff,
        },
        "prediction_cutoff": {
            "kind": cutoff_kind,
            "prediction_timestamp_utc": cutoff.isoformat(),
            "preferred_kind": "post_qualifying",
            "reason": "certified capture"
            if base is not None
            else "no certified post-qualifying capture exists before the cutoff",
        },
        "snapshot": {
            kind: {**variant["snapshot"], "version": LIVE_SNAPSHOT_VERSION}
            for kind, variant in variants.items()
        },
        "schedule_source": schedule_source,
        "roster_source": roster_source,
        "evidence_tier": {
            "training": "Gold",
            "prediction_snapshot": "Development",
            "explanation": (
                "Training rows are audited Gold races. The live snapshot derives from Gold "
                "history, the audited scoring ledger and collector captures, but it is not a "
                "Gold benchmark row until its outcome is audited."
            ),
        },
        "dataset": {
            "path": version.directory.relative_to(root).as_posix(),
            "dataset_version": version.dataset_version,
            "manifest_sha256": version.manifest_sha256,
            "gold_races": len(version.events),
            "gold_rows": len(version.rows),
            "scoring_ledger_sha256": ledger.sha256,
        },
        "dnf_dataset": {
            "path": dnf_version.directory.relative_to(root).as_posix(),
            "dataset_version": dnf_version.dataset_version,
            "manifest_sha256": dnf_version.manifest_sha256,
            "known_dnf_labels": sum(row["label_dnf"] is not None for row in dnf_version.rows),
        },
        "evaluation_protocol": {"version": PROTOCOL["version"], "sha256": PROTOCOL_SHA256},
        "reference_runs": {
            "position": {
                "path": reference_path.relative_to(root).as_posix(),
                "run_id": reference["run_metadata"]["run_id"],
                "task_selection": {
                    task: value["status"]
                    for task, value in reference["task_selection"]["post_qualifying"].items()
                },
            },
            "dnf": {
                "path": dnf_reference_path.relative_to(root).as_posix(),
                "run_id": dnf_reference["run_metadata"]["run_id"],
                "dnf_selection": dnf_reference["task_selection"]["post_qualifying"]["dnf"][
                    "status"
                ],
            },
        },
        "model_choice": {"position": position_choice, "dnf": dnf_choice},
        "models": {
            kind: {
                **record,
                "model_id": model_ids[kind],
                "masked_for_training": list(variants[kind]["masked"]),
                "scoring_gate": variants[kind]["provenance"][0]["scoring_gate"],
                "feature_importance": variants[kind]["importance"],
                "availability_evaluation": variants[kind]["evaluation"],
                "baseline_status": variants[kind]["baselines"]["_status"],
            }
            for kind, record in models.items()
        },
        "missing_features": {kind: variant["missing"] for kind, variant in variants.items()},
        "seeds": {"race": seed, "championship": seed},
        "race_draws": draws,
        "championship": {
            "simulations": simulations,
            "orders_per_session": championship_orders,
            "seed": seed,
            "sha256": championship_sha,
        },
        "execution": {
            "requested_device": device,
            "devices": {
                kind: variant["model"].position.device for kind, variant in variants.items()
            },
            "device_reasons": {
                kind: variant["model"].position.metadata["device_reason"]
                for kind, variant in variants.items()
            },
            "libraries": library_versions(),
        },
        "artifacts": {
            "predictions": {
                "path": predictions_path.relative_to(root).as_posix(),
                "sha256": file_sha256(predictions_path),
            },
            "race_distribution": {"sha256": race_sha},
            "championship": {"sha256": championship_sha},
        },
        "validation_checks": checks,
    }
    manifest_sha = _write_json(run_dir / "manifest.json", manifest)
    report = render_report(manifest, race_payload, championship_payload, table_rows)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "status": STATUS,
        "run_dir": run_dir.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha,
        "event_id": target.event.partition(),
        "cutoff_kind": cutoff_kind,
        "prediction_timestamp_utc": cutoff.isoformat(),
        "report": report,
    }


def validate_publication(
    root: Path,
    *,
    race: list[dict[str, Any]],
    variants: dict[str, dict[str, Any]],
    cutoff: datetime,
    observed_at: datetime,
    version: GoldVersion,
    dnf_version: GoldVersion,
    events: list[EventSimulation],
    model_ids: dict[str, str],
    models: dict[str, dict[str, Any]],
    ledger_sources: list[dict[str, Any]],
    artifacts: list[Path],
) -> dict[str, bool]:
    """Fail closed before a development artifact is reported."""
    _check_race(race)
    if any(not 0 <= row["dnf_probability"] <= 1 for row in race):
        raise ValueError("invalid DNF probability")
    for variant in variants.values():
        for row in variant["rows"]:
            require_known_by(row["feature_timestamp"], cutoff)
            if any(name.startswith("label_") for name in row):
                raise ValueError("live snapshot contains outcome labels")
        snapshot = pq.read_table(root / variant["snapshot"]["path"])
        if file_sha256(root / variant["snapshot"]["path"]) != variant["snapshot"]["sha256"]:
            raise ValueError("frozen snapshot bytes changed")
        if max(snapshot["feature_timestamp"].to_pylist()) > cutoff:
            raise ValueError("frozen snapshot contains post-cutoff information")
        metadata = variant["model"].metadata
        if (
            datetime.fromisoformat(metadata["position_model"]["fit_label_availability_max"])
            > cutoff
        ):
            raise ValueError("position model used post-cutoff labels")
        dnf_max = variant["model"].dnf.metadata["label_available_at_max"]
        if dnf_max is not None and dnf_max > cutoff:
            raise ValueError("DNF model used post-cutoff labels")
    require_known_by(observed_at, cutoff)
    for source in ledger_sources:
        if datetime.fromisoformat(source["effective_at"]) >= cutoff:
            raise ValueError("standings use a points version published after the cutoff")
    for dataset in (version, dnf_version):
        if file_sha256(dataset.directory / "manifest.json") != dataset.manifest_sha256:
            raise ValueError("training dataset manifest changed")
        if dataset.directory.name != dataset.dataset_version:
            raise ValueError("training data does not come from an immutable dataset version")
    allowed = {(model_ids[kind], models[kind]["sha256"]) for kind in model_ids}
    if any((event.model_id, event.source_hash) not in allowed for event in events):
        raise ValueError("championship uses race samples outside this development run")
    for path in artifacts:
        if path.suffix == ".parquet":
            labels = set(pq.read_table(path)["validation_status"].to_pylist())
        else:
            labels = {json.loads(path.read_text(encoding="utf-8"))["validation_status"]}
        if labels != {STATUS}:
            raise ValueError(f"{path.name} is not labelled development_only")
    return {
        "race_probabilities_coherent": True,
        "snapshot_has_no_post_cutoff_data": True,
        "training_uses_immutable_dataset_manifests": True,
        "championship_uses_only_development_race_samples": True,
        "outputs_labelled_development_only": True,
    }
