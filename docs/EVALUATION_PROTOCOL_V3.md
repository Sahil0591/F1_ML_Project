# Gold cutoff-specific evaluation protocol v3

Version 3 (`gold-cutoff-specific-v3`) evaluates development race models separately
for each named weekend cutoff. Version 2 remains the frozen post-qualifying
comparison protocol and its results are unchanged. The settings are frozen in
`src/f1_ml_predictor/prediction/protocol.py`; the canonical SHA-256 of the
protocol dictionary is stored in every v3 evaluation report and prediction
manifest. The protocol was written after the c52b674 forecast was audited and
before any v3 result was used for a live prediction. The persistent-strength
method was revised once, before any live use, after a draft likelihood estimate
proved unidentifiable (see below); that draft output was discarded.

## Cutoff contracts

Each cutoff has its own feature contract (`cutoff-contracts-v1`). History
predictors are recomputed at the contract cutoff from audited Gold outcomes,
audited binary DNF labels and the scoring ledger. Only values published before
the cutoff are read.

| Contract | Historical cutoff | Predictors |
| --- | --- | --- |
| `pre_weekend` | scheduled first practice start | same-season rolling finishes; cross-season driver finish and qualifying form; driver and constructor audited DNF rates; constructor finish, qualifying and teammate form; championship points and positions; circuit history and attrition; whether the circuit was seen before; sprint format |
| `post_practice` | scheduled qualifying start | pre-weekend plus audited FIA practice classification, only when published before qualifying |
| `post_qualifying` | audited Gold post-qualifying cutoff | post-practice plus qualifying position, last-session time and teammate delta |
| `pre_race` | audited Gold post-qualifying cutoff | post-qualifying plus official grids published before the cutoff |

Historical session times come from retained Jolpica season schedules. They place
the cutoff only and are never predictors. A live run hides any predictor that no
driver has at its cutoff from training as well, and is evaluated with the same
predictors hidden.

## Candidates, baselines and folds

Outer folds are complete Gold races in chronological order. A race is used for
training only when every label was published by the test cutoff; audited DNF
labels additionally need their own publication before the cutoff. At least three
earlier races are required.

Joint candidates all produce latent utilities for one coherent sampler:

- `logistic_pl`: logistic winner classifier, utility = log probability. At unit
  temperature without DNF its winner marginals equal the logistic baseline.
- `ridge_pl`: Ridge regression on known finishing positions, utility = minus the
  predicted position.
- `pl_regression`: rank-ordered logit (Plackett-Luce regression) with an L2
  penalty; unknown positions rank after every known finisher and are never
  assigned a place.
- `hist`, `xgboost`, `lightgbm`, `catboost`: the existing boosting position
  regressors.
- `ensemble_linear` and `ensemble_all`: equal mixtures of the calibrated linear
  candidates or of all calibrated candidates.

Baselines are the heuristic and the logistic winner/podium classifiers with Ridge
finish order, fitted on the same contract features. DNF candidates are a smoothed
audited base rate, logistic regression and histogram boosting on reliability
predictors.

## Calibration

Calibration keeps the joint distribution coherent: a race temperature scales a
candidate's utilities, and shrinkage mixes the model with the same sampler at
equal utilities (a DNF-aware uniform race order). Both are chosen from fixed grids
to minimise winner log loss plus race podium log loss divided by three, using only
earlier out-of-fold races (at least five). Sigmoid and isotonic recalibration of a
single marginal were considered and not adopted because they would make winner,
podium and finish probabilities disagree.

For races at circuits absent from earlier Gold races, a separate calibration is
fitted on earlier unseen-circuit races once eight exist. It is used for a race only
if, on earlier unseen races, it already scored better out of fold than shared
calibration. The live run adopts it per candidate on the same evidence.

## Selection

The development primary is chosen prequentially: at each outer race, the model
with the lowest mean calibrated objective on earlier races (after ten races; the
default before that is `logistic_pl`). The resulting selected pipeline has its
own honest out-of-fold score. The live primary is the model with the lowest mean
objective over all out-of-fold races. DNF uses the candidate with the lowest
earlier Brier score.

The formal gate keeps the v2 thresholds per task: 25 paired races, no task
regression against either baseline, and a race-bootstrap 95% interval for the
primary loss (winner log loss, podium Brier, finish MAE) entirely below zero
against both baselines. A provisional result still needs future independent
races.

## Diagnostics

Reports include log loss, Brier score, top-1 and top-3 accuracy, expected
calibration error, reliability curves, ranked probability score, finish MAE,
winner entropy, effective number of win contenders, the fraction of the field
below 0.1% win and 1% podium probability, the maximum win probability, and tail
failures (winners given under 1%, podium finishers given under 1%).

A live forecast is compared with the same model's historical out-of-fold
sharpness at the same cutoff. It is marked `warn` in the sharpest 2.5% of
history and `fail` when sharper than every historical race. This development
quality check is reported separately from mathematical coherence.

## Season uncertainty

Persistent between-race strength is estimated by moment matching. The observed
statistic is the intraclass correlation, within driver-seasons, of standardised
out-of-fold finishing residuals of the best single candidate (classified
positions only, so retirement luck does not enter). The same statistic is
computed on histories simulated from the calibrated model plus driver-season
offsets of each grid standard deviation; the matched value is the persistent
standard deviation, with an interval from a driver-season bootstrap of the
observed correlation. The simulation temperature is refitted with those offsets
so single-race marginals stay calibrated.

A draft Laplace marginal-likelihood estimate was rejected before live use: its
evidence kept rising to the edge of every grid because random offsets also
absorb single-race overdispersion. Offsets that improved single-race scores
showed it was not identifying persistence.

## Addendum season-drift-v1

Season simulations forecast races several rounds ahead with form frozen at the
cutoff. For each historical pre-weekend cutoff race (after calibration warm-up),
the addendum rebuilds pre-weekend rows for the next one to eight races of the same
season using only information known at that cutoff (form frozen; circuit and
format specific), scores them with the strength model fitted for that cutoff and
its calibrated temperature and shrinkage, and selects the per-race random-walk
variance of driver strength that minimises the same calibration objective over
all lags. The next race has lag zero. The addendum has its own SHA-256 so the v3
evaluations remain valid. Its first grid stopped at 0.32 and the estimate landed
on that edge; the grid was extended to 2.25 before the reported Round 16 run, and
the edge run is retained as an earlier development artifact.

Each simulated season draws one world: a random-walk path per driver over the
remaining weekends (plus any persistent offset), shared by every event in that
season. Simulated results do not update rolling form features; updating features
along simulated paths was considered and not adopted, because the validated drift
already measures how far truth moves from a frozen forecast and feature feedback
would double count it.
