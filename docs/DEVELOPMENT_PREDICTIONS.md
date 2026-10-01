# Development predictions for the next race

`predict-next-race` publishes a development prediction for the next Formula 1
race and a development simulation of the rest of the season. Every output is
labelled `development_only`. None of it is a validated forecast.

## Command

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m f1_ml_predictor predict-next-race
```

The command first runs one bounded tick of the existing prospective collector
(`collect-next-race`). The tick records a fresh Jolpica schedule observation and,
once qualifying results are published, freezes a certified post-qualifying
capture. The prediction then:

1. Reads the retained schedule observation, verifies its payload hash and picks
   the first race that starts after the prediction clock.
2. Chooses the cutoff. A certified `post_qualifying` (or later `pre_race`)
   capture for that race is preferred. Without one, the run is labelled
   `pre_qualifying` and its `prediction_timestamp_utc` is the run time.
3. Loads the latest immutable Gold scoring version and the latest audited binary
   DNF version. Every manifest, dataset, coverage and provenance hash is verified.
   The scoring ledger must match the hash bound into the Gold version.
4. Builds one label-free row per entered driver with the same rolling form,
   constructor form and scoring-ledger rules used to build Gold. Only audited
   results and points published before the cutoff are read. The snapshot and its
   source metadata are frozen under `data/features/development_snapshots/`
   before any model is fitted.
5. Fits the development models, samples coherent race outcomes, runs the season
   simulation, validates everything and writes the artifacts.

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--no-collect` | off | Skip the collector tick and use the retained schedule |
| `--season` | next race | Restrict the search for the next race to one season |
| `--simulations` | 100000 | Championship Monte Carlo simulations |
| `--seed` | 42 | Seed for race draws, order samples and the season simulation |
| `--draws` | 65536 | Joint race draws for next-race probabilities (128 to 65536) |
| `--championship-orders` | 8192 | Sampled race orders per remaining session |
| `--device` | auto | `cpu`, `auto` or `cuda`; the existing measured-benefit policy decides |
| `--benchmark-dir`, `--dnf-benchmark-dir` | latest | Pin exact immutable dataset versions |

## Rerunning at later weekend cutoffs

Run the same command again after qualifying. When the collector tick finds a
nonempty qualifying response, it freezes a certified post-qualifying capture and
the new prediction uses it, with the capture's cutoff as `prediction_timestamp_utc`.
If results are not yet published the tick reports `waiting_for_qualifying_results`
and the run stays `pre_qualifying`; try again a few minutes later. For a cutoff
inside the pre-race window, first run
`.\.venv\Scripts\python.exe -m f1_ml_predictor collect-next-race --pre-race`, then
rerun the prediction. Every run gets a new run ID and directory, so earlier
cutoffs are never overwritten.

## Models

No task passes the frozen Gold selection gates, so the run uses development
candidates and says so in its manifest and report:

- **Position strength (winner, podium, finish):** the joint boosting backend
  with the lowest mean rank across winner log loss, podium Brier score and
  finish MAE in the latest frozen comparison for the Gold version. A provisional
  task selection would take precedence. Winner, podium and finish all come from
  one position model so they stay coherent.
- **DNF:** a separate model trained on the audited binary DNF version, using the
  candidate with the lowest observed audited DNF Brier score. DNF therefore does
  not depend on pace. The Gold scoring version has no binary DNF labels.
- **Race distribution:** the existing seeded Plackett-Luce and independent DNF
  sampler. Its temperature is chosen with the frozen grid on the latest
  calibration race, using a DNF model fitted only on labels known at that race.
- **Baselines:** the existing heuristic and logistic/Ridge baselines are fitted on
  the same history and shown next to the model.

Predictors that no driver has at the cutoff, such as qualifying before it has
run, target-weekend practice and point features blocked by the strict ledger
gate, are hidden from the training rows as well. They are never median-filled
at prediction time. A cached diagnostic evaluation repeats the frozen
chronological folds with the same predictors hidden and is reported as model
uncertainty. It is not selection evidence.

## Season simulation

The existing championship simulator resamples whole sampled orders for every
remaining race and sprint. Starting totals are the latest published FIA points
before the cutoff from the audited ledger. Values under appeal are used as
published and named in the report. Races after the next one use the
pre-qualifying model with form held at the cutoff. Rounds after the audited rule
interval continue that season's latest audited tables at full distance. Every
entered driver is assumed points eligible, which keeps the simulator status
`engineering_only`. Outputs include title probabilities, expected final points
and final position distributions for drivers and constructors. Tied positions
are shared evenly; no countback is invented.

## Outputs

Each run writes to
`data/predictions/development/next_race/<event>/<cutoff kind>/<run id>/`:

| File | Content |
| --- | --- |
| `predictions.parquet` | One row per driver: win, podium, DNF, expected finish, distribution, baselines |
| `race_distribution.json` | Full finishing distributions, seed, draws and temperature |
| `championship.json` | Simulator result, standings sources, sessions and assumptions |
| `manifest.json` | Provenance: snapshot, datasets, reference runs, models, devices, checks |
| `report.md` | Readable report |

Fitted models are stored under `models/development/next_race/<run id>/`. Before
the report is written, the run fails closed unless race probabilities are
coherent, the snapshot has nothing after the cutoff, training reads immutable
dataset versions, the season simulation uses only this run's race samples and
every artifact is labelled `development_only`.

## Development only versus validated

A `development_only` output is a reproducible engineering result from the best
current candidate. It has no accuracy claim. Historical outer-fold metrics in the
report show that the logistic baseline still has the lowest observed loss on
winner, podium and finish tasks.

A validated forecast would need a task to pass the frozen Gold protocol (25
paired races, no baseline regression and a race-bootstrap improvement over both
baselines), followed by confirmation on future independent races. Championship
outputs would also need the simulator's validation evidence gate and explicit
points classification. This command never sets those flags.
