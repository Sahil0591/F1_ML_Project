"""Load verified audited history and run cached cutoff-specific evaluations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import joblib

from f1_ml_predictor.prediction.contracts import (
    AuditedHistory,
    build_contract_datasets,
    dnf_label_map,
)
from f1_ml_predictor.prediction.evaluation import (
    analyse,
    collect_out_of_fold,
    simulation_temperature,
)
from f1_ml_predictor.prediction.history import (
    GoldVersion,
    audited_outcomes,
    latest_dnf_directory,
    latest_gold_directory,
    load_gold_version,
)
from f1_ml_predictor.prediction.protocol import (
    CUTOFFS,
    JOINT_CANDIDATES,
    MIN_TRAIN_EVENTS,
    PROTOCOL_V3_SHA256,
)
from f1_ml_predictor.prediction.schedules import retained_schedules
from f1_ml_predictor.scoring.ledger import load_scoring_ledger

COLLECTION_VERSION = "oof-fits-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_audited_history(
    root: Path,
    *,
    gold_dir: Path | None = None,
    dnf_dir: Path | None = None,
    http_client: httpx.Client | None = None,
) -> tuple[AuditedHistory, GoldVersion, dict[str, dict[str, str]]]:
    """Verify Gold, DNF, ledger and schedule inputs before any feature is computed."""
    version = load_gold_version(gold_dir or latest_gold_directory(root))
    dnf_version = load_gold_version(dnf_dir or latest_dnf_directory(root))
    outcomes = audited_outcomes(root, version)
    ledger = load_scoring_ledger(
        root / "data/audit/scoring_rules.json", root / "data/audit/event_points_evidence.json"
    )
    if ledger.sha256 != version.manifest.get("scoring_ledger_sha256"):
        raise ValueError("scoring ledger differs from the Gold version; rebuild Gold scoring first")
    seasons = sorted({int(row["event_id"][7:11]) for row in version.rows})
    weekends, sources = retained_schedules(root, seasons, http_client=http_client)
    history = AuditedHistory(version, outcomes, ledger, weekends, dnf_label_map(dnf_version))
    return history, dnf_version, sources


def evaluate_contract(
    root: Path,
    history: AuditedHistory,
    contract: str,
    rows: list[dict[str, Any]],
    dataset_sha256: str,
    *,
    masked: tuple[str, ...] = (),
    seed: int = 42,
    candidates: tuple[str, ...] = JOINT_CANDIDATES,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run or reload the frozen v3 evaluation for one contract and availability mask."""
    key_payload = {
        "contract": contract,
        "dataset_sha256": dataset_sha256,
        "masked_features": sorted(masked),
        "seed": seed,
        "candidates": list(candidates),
        "protocol_sha256": PROTOCOL_V3_SHA256,
        "device": "cpu",
    }
    key = hashlib.sha256(_canonical(key_payload)).hexdigest()
    path = (
        root
        / "models/experiments/gold"
        / history.version.dataset_version
        / "cutoff_v3"
        / contract
        / key
        / "evaluation.json"
    )
    if path.exists():
        cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("key") != key_payload:
            raise ValueError("cached cutoff evaluation does not match its inputs")
        return cached, path
    oof_payload = {
        "contract": contract,
        "dataset_sha256": dataset_sha256,
        "masked_features": sorted(masked),
        "seed": seed,
        "candidates": list(candidates),
        "minimum_train_events": MIN_TRAIN_EVENTS,
        "collection_version": COLLECTION_VERSION,
    }
    oof_key = hashlib.sha256(_canonical(oof_payload)).hexdigest()
    oof_path = path.parents[1] / "oof" / f"{oof_key}.joblib"
    if oof_path.exists():
        records = joblib.load(oof_path)
    else:
        records = collect_out_of_fold(
            rows,
            contract,
            masked=masked,
            seed=seed,
            device="cpu",
            candidates=candidates,
            progress=progress,
        )
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(records, oof_path)
    report = analyse(records, seed=seed, candidates=candidates)
    live = report["live"]
    strength = live["strength_model_for_uncertainty"]
    sigma = live["persistent_strength"]["persistent_standard_deviation"]
    live["simulation_temperature"] = simulation_temperature(
        records,
        strength,
        report["dnf"]["prequential_choices"],
        live["calibration"][strength]["shrink"],
        sigma,
        seed=seed,
    )
    report["key"] = key_payload
    report["devices"] = sorted({device for record in records for device in record.devices.values()})
    report["gold_dataset_version"] = history.version.dataset_version
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode())
    return report, path


def evaluate_cutoffs(
    root: Path,
    *,
    contracts: tuple[str, ...] = CUTOFFS,
    seed: int = 42,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build every contract dataset and evaluate it under protocol v3."""
    history, dnf_version, sources = load_audited_history(root)
    datasets = build_contract_datasets(root, history, dnf_version, sources)
    summary = {}
    for contract in contracts:
        dataset = datasets[contract]
        report, path = evaluate_contract(
            root,
            history,
            contract,
            dataset["rows"],
            dataset["manifest"]["dataset_sha256"],
            seed=seed,
            progress=progress,
        )
        summary[contract] = {
            "path": path.relative_to(root).as_posix(),
            "primary": report["live"]["primary"],
            "dnf_model": report["live"]["dnf_model"],
            "outer_races": report["outer_races"],
            "formal_gate": {
                name: {task: value["status"] for task, value in result["formal_gate"].items()}
                for name, result in report["models"].items()
            },
        }
    return summary
