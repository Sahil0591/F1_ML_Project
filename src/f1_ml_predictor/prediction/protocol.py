"""Frozen cutoff-specific evaluation protocol for development race models (v3).

Version 2 remains the post-qualifying Gold comparison protocol. Version 3 adds
separate feature contracts, chronological evaluation, calibration and selection
for each named weekend cutoff. The settings below are fixed before evaluation.
"""

import hashlib
import json

from f1_ml_predictor.models.protocol import PRELIMINARY_PAIRED_EVENTS, SELECTION_PAIRED_EVENTS

PROTOCOL_V3_VERSION = "gold-cutoff-specific-v3"
CUTOFFS = ("pre_weekend", "post_practice", "post_qualifying", "pre_race")
JOINT_CANDIDATES = (
    "logistic_pl",
    "ridge_pl",
    "pl_regression",
    "hist",
    "xgboost",
    "lightgbm",
    "catboost",
)
ENSEMBLES = ("ensemble_linear", "ensemble_all")
DNF_CANDIDATES = ("prior", "logistic", "hist")
TEMPERATURES = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)
SHRINKAGE = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
MIN_TRAIN_EVENTS = 3
MIN_CALIBRATION_EVENTS = 5
MIN_SELECTION_EVENTS = 10
MIN_UNSEEN_CALIBRATION_EVENTS = 8
DEFAULT_CANDIDATE = "logistic_pl"
CALIBRATION_DRAWS = 2048
PERSISTENT_SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)

PROTOCOL_V3 = {
    "version": PROTOCOL_V3_VERSION,
    "tier": "Gold",
    "cutoffs": list(CUTOFFS),
    "cutoff_times": {
        "pre_weekend": "scheduled first practice start",
        "post_practice": "scheduled qualifying start; practice only if published before it",
        "post_qualifying": "audited Gold post-qualifying cutoff",
        "pre_race": "audited Gold post-qualifying cutoff plus grids published before it",
    },
    "outer_split": "complete events in chronological rolling origins, one cutoff at a time",
    "label_gate": "every training label available by the test cutoff",
    "minimum_train_events": MIN_TRAIN_EVENTS,
    "joint_candidates": list(JOINT_CANDIDATES),
    "ensembles": list(ENSEMBLES),
    "baselines": ["heuristic", "logistic"],
    "dnf_candidates": list(DNF_CANDIDATES),
    "calibration": {
        "family": "race temperature on candidate utilities and mixture with a DNF-aware "
        "uniform race prior; both keep the joint distribution coherent",
        "temperatures": list(TEMPERATURES),
        "shrinkage": list(SHRINKAGE),
        "fit": "prequential: earlier out-of-fold races only",
        "minimum_events": MIN_CALIBRATION_EVENTS,
        "unseen_circuit_rule": "separate parameters fitted on earlier unseen-circuit races "
        f"once {MIN_UNSEEN_CALIBRATION_EVENTS} exist; adopted only if better out of fold",
        "objective": "winner log loss plus race podium log loss divided by three",
        "draws": CALIBRATION_DRAWS,
    },
    "selection": {
        "development_primary": "prequential: at each race the candidate with the lowest mean "
        f"calibrated objective on earlier races once {MIN_SELECTION_EVENTS} exist, otherwise "
        f"{DEFAULT_CANDIDATE}; the live primary uses all out-of-fold races",
        "formal_gate": f"{SELECTION_PAIRED_EVENTS} paired races, no task baseline regression and "
        "a race-bootstrap improvement excluding zero against both baselines",
        "preliminary_paired_events": PRELIMINARY_PAIRED_EVENTS,
    },
    "season_uncertainty": {
        "persistent_strength": "driver-season offset SD whose simulated histories match the "
        "observed intraclass correlation of standardized out-of-fold finishing residuals "
        "(classified positions only) for the best single candidate",
        "sigma_grid": list(PERSISTENT_SIGMAS),
        "simulation_temperature": "temperature refitted with those offsets so single-race "
        "marginals stay calibrated",
    },
    "confirmation": "future independent races required after any selection",
}
PROTOCOL_V3_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL_V3, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

# Addendum for the race prediction made once the FIA provisional starting grid is
# out. Protocol v3 and its four cutoffs stay frozen; this contract is evaluated
# with the same method and stored under its own key.
GRID_CONTRACT = "provisional_grid"
GRID_ADDENDUM = {
    "version": "grid-addendum-v1",
    "base_protocol": PROTOCOL_V3_VERSION,
    "base_protocol_sha256": PROTOCOL_V3_SHA256,
    "tier": "Gold",
    "contract": GRID_CONTRACT,
    "cutoff": "one minute after the first FIA Grand Prix starting grid publication in the "
    "retained registry, never before the audited Gold post-qualifying cutoff; a race whose "
    "grid cannot be audited keeps that cutoff with grid inputs missing",
    "grid_inputs": {
        "grid_position": "ordinal printed on the FIA grid known at the cutoff; missing for a "
        "pit-lane start",
        "grid_places_lost": "grid position minus qualifying position; for a pit-lane start, "
        "entrants plus one minus qualifying position",
        "pit_lane_start": "1 when the grid names the driver as a pit-lane starter, else 0",
    },
    "other_inputs": "the pre_race contract predictors, history recomputed at this cutoff",
    "evaluation": "protocol v3 outer split, label gate, candidates, calibration and gate",
    "selection": "protocol v3 formal gate; no change to the frozen v3 cutoffs",
}
GRID_ADDENDUM_SHA256 = hashlib.sha256(
    json.dumps(GRID_ADDENDUM, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
CONTRACTS = (*CUTOFFS, GRID_CONTRACT)
