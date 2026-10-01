# Gold chronological evaluation protocol v2

Version: `gold-chronological-v2`. The source constant and JSON digest in
`src/f1_ml_predictor/models/protocol.py` identify the machine-readable contract.
Version 1 remains available for interpreting earlier reports.

Each named prediction cutoff is evaluated separately. All drivers from an event
stay in one outer fold, ordered by prediction cutoff. Fit labels must be available
before the calibration cutoff, and calibration labels before the outer test cutoff.
Preprocessing, estimator fitting, DNF calibration and race-strength temperature
selection use only those earlier folds. A model never fits or calibrates on its
test event. All backends and baselines are compared on identical outer event and
driver observations.

Preliminary comparison requires at least 15 paired Gold races. Provisional
selection for each task requires at least 25 eligible Gold races and 25 paired
outer test races. Winner, podium, binary DNF and finishing position are gated
independently. A task needs its own evaluated metrics, no recorded regression
against either heuristic or logistic baseline on that task, and a race-bootstrap
95% upper bound below zero for its primary loss difference against both baselines.
Primary losses are winner categorical log loss, podium Brier, binary DNF Brier
and finishing position MAE. Winner Brier and top-1/top-3 accuracy, podium and DNF
log loss, and rank diagnostics remain secondary regression checks. A passing task
chooses the lowest primary loss among passing backends. Otherwise it records
`no_selection` with per-backend reasons.

An observed lowest loss is descriptive, not a selection. Feature ablations and
calibration variants retain distinct run IDs and report `diagnostic_only` where
applicable. Comparing several variants using the same outer folds is exploratory;
a candidate chosen from those folds must be frozen and confirmed on future
independent races before a validated prediction claim. Historical Gold eligibility
may improve, but a new benchmark manifest produces a new dataset version and run.
Future outcomes used to confirm one frozen candidate cannot tune that candidate.

The joint race sampler must produce exactly one winner, three podium positions,
one occupant per modeled finish position, and driver distributions summing to one.
DNF probabilities remain separate from race pace and require audited binary labels.
Historical held-out development predictions may be published without labels and
must be marked `development_only`. Validated WDC and WCC outputs remain gated on
prospective race model confirmation.
