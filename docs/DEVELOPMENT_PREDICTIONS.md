# Development predictions for the next race

`predict-next-race` publishes a development prediction for the next Formula 1
race and a development simulation of the rest of the season. Every output is
labelled `development_only`. None of it is a validated forecast.

## Commands

```powershell
.\.venv\Scripts\python.exe -m f1_ml_predictor evaluate-cutoffs
.\.venv\Scripts\python.exe -m f1_ml_predictor predict-next-race
```

`evaluate-cutoffs` builds the four cutoff-contract datasets and runs the frozen
[protocol v3](EVALUATION_PROTOCOL_V3.md) evaluation for each. Results are cached by
their exact inputs under `models/experiments/gold/<dataset>/cutoff_v3/`, so later
runs reuse them. `predict-next-race` also triggers any evaluation it needs.

The prediction command first runs one bounded tick of the existing prospective
collector. It then:

1. Picks the first race after the prediction clock from the fresh, hash-verified
   schedule observation. The circuit comes from Jolpica's `circuitId`, never from
   the event title, so "Bahrain Grand Prix in Malaysia" resolves to `sepang`.
2. Chooses the cutoff contract. A certified post-qualifying (or pre-race) capture
   selects `post_qualifying` (or `pre_race`). Without one the run uses
   `pre_weekend`; if weekend sessions have already run but were not captured, the
   report says so.
3. Loads the latest immutable Gold scoring version, the latest audited binary DNF
   version, the scoring ledger and retained season schedules, verifying every hash.
4. Builds one label-free row per driver with the same contract code used for the
   historical datasets, from audited data published before the cutoff only.
5. Hides predictors that no driver has at the cutoff (for example points features
   blocked by the strict ledger gate) from training as well, and uses the contract's
   v3 evaluation repeated with exactly those predictors hidden. Circuit history is
   exempt: at an unseen circuit its missingness is structural and its flags are
   well represented in training.
6. Fits every candidate, the selected DNF model and the baselines on all complete
   Gold races before the cutoff, then samples the race with the calibrated primary.
7. Runs the season simulation, validation and quality checks, and writes artifacts.

Options: `--no-collect`, `--season`, `--simulations` (default 100000), `--seed`
(42), `--draws` (65536 race draws), `--worlds` (1000 model-uncertainty worlds),
`--orders-per-world` (16), `--device` (`cpu`, `auto`, `cuda`; the measured-benefit
GPU policy decides), `--benchmark-dir`, `--dnf-benchmark-dir`, and
`--methodology c52b674` to rerun the earlier single-model pipeline unchanged.

## Rerunning at later weekend cutoffs

Run `predict-next-race` again after qualifying. When the collector tick finds a
nonempty qualifying response it freezes a certified capture and the new run uses
the `post_qualifying` contract with its own evaluation, calibration and selection.
The pre-qualifying primary is never reused after qualifying. If results are not yet
published, the tick reports `waiting_for_qualifying_results`; retry a few minutes
later. For the pre-race window run
`.\.venv\Scripts\python.exe -m f1_ml_predictor collect-next-race --pre-race` first.
Practice is not captured live yet, so a run between practice and qualifying uses
the `pre_weekend` contract and says so. Every run has a new run ID and directory.

## Models and calibration

Each contract selects its own primary from the joint candidates (logistic,
Ridge and Plackett-Luce strengths, four boosting backends and two ensembles).
Baseline-derived candidates drive the same coherent sampler as boosting, so a
baseline is never passed over because the simulator expects boosting. The
logistic and Ridge baselines and the best single boosting model are reported
next to the primary. Calibration is a race temperature plus mixing with a
DNF-aware uniform race order, fitted only on earlier out-of-fold races, so winner,
podium, finish and DNF stay coherent. No probability floor is imposed.

DNF uses the candidate with the best out-of-fold Brier score on audited binary
labels (1,506 after the [v2 expansion](DNF_AUDIT_STATUS.md)). If that is the base
rate, every driver gets the same DNF probability and the report labels it as a
field-wide rate rather than an individualised prediction.

## Checks reported separately

- **Coherence and leakage** (fail closed): race probabilities are coherent, no
  snapshot value or training label is later than the cutoff, training reads
  immutable dataset versions, the season uses only this run's race samples, and
  every artifact is labelled `development_only`.
- **Development quality** (`pass`, `warn` or `fail`): live sharpness compared with
  the same model's historical out-of-fold forecasts at the same cutoff, including
  winner entropy, effective number of win contenders, the share of the field below
  0.1% win or 1% podium probability, and the maximum win probability.
- **Out of distribution**: unseen circuit, feature ranges, missing-value patterns
  and nearest-row distance against leave-one-race-out training distances. Unseen
  circuits use a separate calibration only where it beat shared calibration out of
  fold; no circuit history is invented.

## Season simulation

Every remaining race and sprint gets its own pre-weekend rows: circuit history,
circuit attrition and sprint format change by event, while driver form is the
form known at the cutoff. Model uncertainty is represented by worlds. Each world
draws a driver strength random walk across the remaining weekends with per-race
variance validated on historical stale-form forecasts (protocol addendum
`season-drift-v1`), plus any persistent offset supported by out-of-fold residual
correlation. All events in one simulated season share that world, so uncertainty
does not average away across races. The simulator resamples whole joint orders
within the world and updates points after every event.

The number of simulations controls Monte Carlo noise only. Title probabilities
are reported to whole percentage points, with Monte Carlo error bounded by the
number of worlds. The report also shows titles with strength fixed (the c52b674
assumption) and under single candidate models.

## Outputs

Each run writes to
`data/predictions/development/next_race/<event>/<cutoff>/<run id>/`:
`predictions.parquet`, `race_distribution.json` (primary, calibration stages,
candidates and baselines), `championship.json`, `manifest.json` (provenance,
evaluation, OOD, sharpness, checks), `comparison.json` (against the c52b674
regression fixture when the event matches) and `report.md`. Fitted models are in
`models/development/next_race/<run id>/`. The c52b674 forecast is preserved in
`docs/regression/c52b674-2026-round16-pre-qualifying.json`. The legacy pipeline resolves the latest DNF
version, which is now the 95-race expansion; an exact c52b674 reproduction pins
the original 29-race DNF version recorded in that fixture.

## Development only versus validated

A `development_only` output is the best current candidate under a frozen protocol
with honest out-of-fold evidence. A task is formally selected only when it beats
both baselines on 25 paired races with a race-bootstrap interval excluding zero
and no regressions, and even then it needs future independent races before any
forecast is called validated. Championship outputs would also need the
simulator's validation evidence and explicit points classification. This command
never sets those flags.

## Sprint weekends

On a sprint weekend `predict-next-race` publishes a sprint race prediction between
sprint qualifying and the sprint, and the Grand Prix prediction otherwise
(`--session auto`, the default; `--session race` or `--session sprint` forces one).

1. After sprint qualifying, run `predict-next-race`. Its collector tick freezes the
   OpenF1 sprint qualifying result once it is nonempty, mapped to canonical drivers
   through the session's own driver list and the Jolpica season codes. The run uses
   the `post_sprint_qualifying` cutoff and addendum
   [sprint-dev-v1](EVALUATION_PROTOCOL_V3.md#addendum-sprint-dev-v1), and writes to
   `data/predictions/development/next_race/<event>/post_sprint_qualifying/<run id>/`.
   A driver who set no sprint qualifying time keeps a missing grid position. Each
   driver's constructor is the team in the capture's own OpenF1 driver list, so a
   mid-season seat change is followed; the roster source names drivers whose team
   differs from the latest audited roster. An unmapped team name falls back to the
   latest audited roster, then to the driver's latest Jolpica entry this season. A
   Grand Prix run before the Grand Prix capture (the pre-weekend contract) uses the
   same captured roster.
2. After the sprint, run `collect-next-race`. It freezes the Jolpica sprint
   classification (polled from 45 minutes after the scheduled sprint start), then
   the FIA Final Sprint Classification once it is published with no later sprint
   ruling pending, reached through the FIA registry for the exact season and event
   and rechecked after the PDF is read. Later Grand Prix runs add the sprint points
   to the starting standings: from the FIA classification and the audited sprint
   table when it is captured (the notes say whether Jolpica agreed), otherwise from
   Jolpica, marked Development. The season simulation no longer samples that
   sprint. Without the Jolpica capture a Grand Prix run after the sprint stops
   rather than drop the points; the FIA capture never blocks a run.

Sprint captures are immutable, hash-named bundles under
`data/raw/prospective_sprint/<event>/`; the observation clock is the availability
time. Sprint history is the FIA-audited Gold sprint version when one exists
(`build-gold-sprints`, addendum
[sprint-gold-v2](EVALUATION_PROTOCOL_V3.md#addendum-sprint-gold-v2)), and the
Development history otherwise; the manifest records which. Grand Prix runs at the
post-qualifying and pre-race cutoffs use the captured sprint qualifying position,
sprint position and classified flag (contract `cutoff-contracts-v3`), trained on the
Gold sprint history. Captured sprint points must match the season's audited sprint
points table or the run stops.
