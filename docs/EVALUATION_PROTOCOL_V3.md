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

Each cutoff has its own feature contract (`cutoff-contracts-v3`; v3 adds same-weekend
sprint values to the post-qualifying contracts, v2 added the strength
features below to v1). History
predictors are recomputed at the contract cutoff from audited Gold outcomes,
audited binary DNF labels and the scoring ledger. Only values published before
the cutoff are read.

| Contract | Historical cutoff | Predictors |
| --- | --- | --- |
| `pre_weekend` | scheduled first practice start | same-season rolling finishes; cross-season driver finish and qualifying form; driver and constructor audited DNF rates; constructor finish, qualifying and teammate form; championship points and positions; circuit history and attrition; whether the circuit was seen before; sprint format; pairwise driver and constructor Elo; teammate qualifying head-to-head; similarity-weighted driver and constructor finish deltas at the most similar profiled circuits |
| `post_practice` | scheduled qualifying start | pre-weekend plus audited FIA practice classification, only when published before qualifying |
| `post_qualifying` | audited Gold post-qualifying cutoff | post-practice plus qualifying position, last-session time and teammate delta; on sprint weekends the Gold sprint qualifying position, sprint position and whether the driver was classified, each only when published before the cutoff (v3) |
| `pre_race` | audited Gold post-qualifying cutoff | post-qualifying plus official grids published before the cutoff |

Historical session times come from retained Jolpica season schedules. They place
the cutoff only and are never predictors. A live run hides any predictor that no
driver has at its cutoff from training as well, and is evaluated with the same
predictors hidden.

### Strength features (v2)

Adapted from open-source F1 predictors and computed from the same audited
history, so they obey the same cutoff rules:

- `driver_elo`, `constructor_elo`, `driver_elo_events`: pairwise Elo over
  classified, non-retired finishers (after Malek1414/f1-predictions). Expected
  scores use driver plus constructor rating; team-mate pairs count double for
  drivers and not at all for constructors. Ratings regress towards 1500 between
  seasons (drivers keep 75%, constructors 50%, or 20% in regulation-reset
  seasons). A retirement is only excluded once its audited DNF label is
  published before the cutoff.
- `driver_teammate_qualifying_h2h_10`: share of the latest ten appearances in
  which the driver out-qualified the team-mate (after MynosIII/F1Predictor).
- `driver_similar_circuit_delta`, `constructor_similar_circuit_delta`: finish
  relative to the entity's own recent mean at the five most similar circuits,
  weighted by similarity of a hand-curated profile (length, turns, street
  circuit, downforce level) in `prediction/strength_features.py` (after
  MynosIII/F1Predictor). This gives unseen circuits such as Sepang a track-type
  signal. Circuits without a profile leave these features missing with reason
  `circuit_unprofiled`.

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

## Addendum sprint-dev-v1

Sprint races are predicted at a separate cutoff, `post_sprint_qualifying`. The
strength candidates are the v3 joint candidates fitted on Gold `post_qualifying`
race rows whose labels were published by the sprint cutoff. A sprint row is the
same race contract row with the sprint qualifying classification in the
qualifying fields (position, lap in the last stage reached, teammate delta);
practice predictors are hidden because practice is not captured live.

Sprints are shorter and more grid-bound than races, so calibration is not reused
from races. The unchanged v3 `analyse` runs over sprint out-of-fold records only:
temperature, prior mixing and the development primary are prequential on earlier
sprints. The sprint retirement prior is the smoothed rate of earlier sprints; the
race-trained logistic and histogram DNF models compete with it on out-of-fold
sprint Brier score. Baselines are the v3 heuristic and logistic models fed the
sprint rows, and the formal gate is unchanged.

The sprint history is Development tier. Labels come from the Jolpica sprint
classification; the 2023+ sprint grid comes from OpenF1 sprint qualifying (sprint
shootout in 2023), and the 2022 grid from Friday qualifying, which set the sprint
grid that season. Every Jolpica sprint points value must equal the FIA sprint
points in the audited scoring ledger and every FIA sprint position must match;
otherwise the sprint is excluded with its reason. The addendum has its own
SHA-256, and the frozen v3 race evaluations are unaffected.

First evaluation (7 October 2026): 25 sprints assembled, one excluded (2023 Qatar,
where Jolpica and the FIA after-sprint points disagree on Stroll's position), 24
outer sprints. The primary (`ridge_pl`) had winner log loss 1.160 against 1.406
(logistic) and 1.628 (grid heuristic), podium Brier 0.0564 against 0.0680 and
0.0653, and finish MAE 2.23 against 2.30 and 2.32. The grid heuristic picked the
winner more often (top-1 0.63 against 0.46), so every task stays `no_selection`
and sprint predictions are development only. A full FIA Gold audit of sprint
classifications is planned to replace the Development history.

## Addendum sprint-gold-v1

`build-gold-sprints` audits sprint history directly from FIA documents
(`trust/sprint_gold.py`). For each completed sprint weekend it reads the retained
FIA event registry and:

- takes the sprint grid from the first non-recalled sprint qualifying
  classification (2024+), sprint shootout classification (2023) or, in 2022, the
  main qualifying classification, which set the sprint grid. Its availability is the
  registry publication clock read as the later UTC bound plus one minute;
- requires that grid to be published before the earliest sprint starting grid,
  which the FIA issues before the start;
- labels from the latest non-recalled Final Sprint Classification, and excludes the
  weekend if a later sprint ruling (decision, infringement, penalty, review) could
  amend it. Late re-uploads of grid documents and championship points are not
  rulings;
- checks every classified position against the audited FIA sprint points, and
  labels retirements only on FIA, Jolpica and OpenF1 agreement under the unchanged
  binary DNF rule. A classified driver who set no sprint qualifying time keeps a
  missing grid position; a grid driver missing from the classification excludes the
  weekend.

First version (7 October 2026): 23 of 26 sprints, 470 driver rows, 402 audited
retirement labels. Excluded: 2023 Azerbaijan (a grid driver is absent from the
classification), 2025 China (the PDF layout splits a driver name, so the table is not
parsed) and 2025 São Paulo (a corrected sprint infringement after the final
classification). 2023 Qatar, excluded by the Development cross-check, is included:
the FIA classification agrees with the audited FIA points, and Jolpica differed.

Sprint calibration (addendum `sprint-dev-v1` settings) now runs on this Gold history
when a Gold sprint version exists, as `sprint-gold-v1` with its own SHA-256. On 22
outer sprints with the v2 race contract the primary (`ridge_pl`) had winner log loss
1.313 against 1.572 (logistic) and 1.628 (grid heuristic), podium Brier 0.0599
against 0.0673 and 0.0667, and finish MAE 2.34 against 2.35 and 2.32. The heuristic
again picked more winners, so sprint tasks stay `no_selection`. With 402 labels the
logistic retirement model beat the flat sprint rate out of fold.

## Addendum sprint-gold-v2

The same method as `sprint-gold-v1` on audit method `fia-sprint-direct-v2`, which
recovers two of the three sprints v1 excluded (8 October 2026):

- 2025 China: the FIA sprint qualifying PDF wraps "Andrea Kimi ANTONELLI" onto two
  lines. A lone uppercase surname is rejoined only when the row's name cell plus
  that word is an exact driver alias, written into the cell padding so the
  lap-time columns keep their positions.
- 2025 São Paulo: document 65 is a corrected reissue of recalled document 41 with
  the same 5 second penalty for car 30, and the Final Sprint Classification
  (document 42) already applies it and cites document 41. A later sprint ruling
  reviewed this way is listed with its finding and bound to the ruling PDF hash; a
  changed PDF excludes the sprint again.

The label rule therefore reads "no later sprint ruling, except one reviewed against
the classification and found not to amend it", so the addendum has a new version
and SHA-256. The rebuilt Gold sprint set has 25 of 26 sprints, 510 driver rows and
442 audited retirement labels; 2023 Azerbaijan stays excluded.

On 24 outer sprints the primary (`ridge_pl`) had winner log loss 1.300 against 1.580
(logistic) and 1.599 (grid heuristic), podium Brier 0.0609 against 0.0674 and
0.0653, and finish MAE 2.33 against 2.31 and 2.34. The heuristic picked the winner
more often (top-1 0.67 against 0.46), so sprint tasks stay `no_selection`. The
logistic retirement model again beat the flat sprint rate (Brier 0.0601 against
0.0638).

The race contracts carry Gold sprint values, so both new sprints change the
`post_qualifying` and `pre_race` datasets. Re-evaluated on the same 92 outer races,
the `ensemble_all` primary moved from winner log loss 1.2245 to 1.2230 at
`post_qualifying` (podium Brier 0.07126 to 0.07123, finish MAE 2.641 to 2.643) and
from 1.2247 to 1.2251 at `pre_race` (0.07135 to 0.07121, 2.637 to 2.638). Formal
gates were identical.

### Sprint values in the race contracts (cutoff-contracts-v3)

Since 2024 the sprint runs before Grand Prix qualifying, so the `post_qualifying`
and `pre_race` contracts add the Gold sprint qualifying position, sprint position and
a classified flag, each gated on its own publication clock. Earlier formats and
non-sprint weekends leave them missing; the sprint model hides them because they
describe the sprint it predicts. A separate post-sprint race cutoff was not added: it
would change the frozen v3 cutoff list for a window of a few hours.

The paired check used the same v3 dataset with the sprint values hidden (the v2
feature set), on the same 92 outer races, folds and seed. Thirteen evaluated sprint
weekends carry values. The `ensemble_all` primary moved from winner log loss 1.2315
to 1.2245, podium Brier 0.07135 to 0.07126 and finish MAE 2.643 to 2.641 at
`post_qualifying`, and from 1.2276 to 1.2247, 0.07125 to 0.07135 and 2.640 to 2.637
at `pre_race`. Formal gates were identical. The changes are within noise for 13
weekends; v3 was kept because it was never materially worse.
