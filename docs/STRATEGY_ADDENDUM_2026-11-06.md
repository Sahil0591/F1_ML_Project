# Strategy addendum (2026-11-06)

Tests whether tyre, pit-stop and practice long-run evidence improves the
post-qualifying prediction, alongside the weather forecast. It does not, so live
predictions are unchanged and the contract runs as a shadow.

## Evidence

`scripts/audit-strategy-history.py <coverage report>` freezes OpenF1 historical
stints, pit stops, laps and drivers for each Gold race from 2023 (OpenF1 starts
then) and its practice sessions. OpenF1 serves completed sessions free from 30
minutes after they end but does not certify publication times, so:

- a race counts from 24 hours after its scheduled end;
- a practice session counts from 2 hours after its end.

Requests are retained under `data/raw/openf1_strategy` and reused, so a run
resumes where it stopped; OpenF1's "no results" 404 is retained as an empty
session. Summaries are replayed from the retained bytes and cached per audit,
roster set and summary code version.

The audit `strategy-audit-5a371ce6...` covers 83 of 99 Gold races (16 are 2022).
Car numbers map to Gold IDs by exact FIA name, then by a unique surname for a
driver no exact name claimed (a reserve such as Arthur Leclerc never becomes
Charles). Pit-lane times are measured against the race's normal-stop cluster;
red-flag waits and repairs are dropped. Eight races used intermediates or wets,
among them Brazil 2024.

## Contract

`post_qualifying_strategy` is addendum `strategy-addendum-v1` on top of the
weather addendum, adding:

| Predictor | Driver rows |
| --- | ---: |
| `driver_tyre_degradation_10` | 1,664 / 2,015 |
| `constructor_tyre_degradation_10` | 1,667 |
| `constructor_pit_lane_delta_10` | 1,567 |
| `circuit_median_stops` | 1,172 |
| `circuit_pit_lane_seconds` | 1,072 |
| `practice_long_run_gap` | 1,231 |
| `practice_long_run_degradation` | 996 |

## Result

Against `post_qualifying` on 96 paired races (strategy minus base, 95% race
bootstrap):

| Model | Winner log loss | Podium Brier | Position MAE |
| --- | ---: | ---: | ---: |
| ensemble_all | +0.031 [+0.009, +0.054] | +0.0001 [-0.0008, +0.0012] | +0.022 [+0.010, +0.035] |
| ensemble_all, practice only | +0.015 [-0.003, +0.033] | +0.0001 [-0.0007, +0.0012] | +0.019 [+0.009, +0.030] |
| ensemble_all, history only | +0.014 [-0.004, +0.033] | +0.0000 [-0.0006, +0.0007] | +0.009 [-0.001, +0.020] |
| catboost | -0.033 [-0.093, +0.031] | +0.0012 [-0.0007, +0.0032] | +0.009 [-0.011, +0.028] |
| ridge_pl | +0.115 [+0.020, +0.240] | +0.0018 [-0.0015, +0.0068] | +0.045 [+0.014, +0.086] |

The full contract is worse for the ensemble, mostly through the linear members;
neither group helps alone. Qualifying and existing form already carry most of
what long runs and tyre history say.

## Wet labels for the wet-skill shadow

Fitted tyres are a better wet label than a forecast: 2026 round 4 had 14 mm
forecast but no intermediates. The wet-skill shadow now learns skills from races
that used intermediates or wets (forecast labels only before OpenF1), and still
switches on from the live forecast. Out of fold on the 18 forecast-wet races it
improves winner log loss by 0.038 [-0.091, +0.019] against the pooled model.
Counting wins, Verstappen won 4 of the 8 wet races against 1.9 expected, and 32
of 75 dry races since 2023 against 20.9.

## Shadows

`predict-next-race` publishes a `strategy` shadow, with this weekend's practice
fetched from OpenF1 after the 2 hour margin. Re-run
`scripts/audit-strategy-history.py` after each new Gold race, or the shadow is
skipped.
