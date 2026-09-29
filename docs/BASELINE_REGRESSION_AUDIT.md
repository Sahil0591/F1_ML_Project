# Gold baseline regression audit, 2026-09-29

The [frozen protocol](EVALUATION_PROTOCOL_V1.md) was rerun without changing
folds, metrics, selection rules or model configurations. Four additional 2026
Gold races passed the existing exact FIA and schedule audit: Azerbaijan, Hungary,
Belgium and Great Britain. Italy and the Netherlands were excluded because later
event documents need final-outcome review. The cohort now has 29 races, 594
driver-race observations and 24 unique drivers. The four backends and paired
baselines share 26 test races and 534 driver rows. DNF has 494 known test labels;
finish-position error has 466 known ordinals. All fits used CPU.

## Frozen result

| Model | Winner log loss | Podium log loss | DNF log loss | Position MAE |
| --- | ---: | ---: | ---: | ---: |
| Logistic baseline | 1.274 | 0.372 | 0.438 | 2.788 |
| CatBoost | 1.025 | 0.399 | 0.400 | 2.748 |
| Histogram boosting | 1.070 | 0.440 | 0.416 | 2.927 |
| LightGBM | 1.207 | 0.436 | 0.408 | 2.898 |
| XGBoost | 1.298 | 0.445 | 0.408 | 2.986 |

The official pooled metrics and complete protocol regression lists are in the
versioned local comparison report. Every backend retains at least one recorded
baseline regression. CatBoost improves winner log loss but regresses against
logistic podium log loss by 0.027 in the protocol report. Histogram boosting,
LightGBM and XGBoost regress on both podium log loss and Brier score, and on
position MAE. XGBoost also regresses on winner log loss. The selection state is
`deferred` with no selected backend despite reaching 26 paired races.

## Paired race diagnostics

The separate `regression_audit.json` uses each race as one unit. It contains
every fold's loss delta, the five largest failures for each task, and 5,000
race-bootstrap resamples with seed 42. A positive delta means the backend is
worse than logistic. The intervals describe variation across these 26 races;
they are exploratory and do not account for multiple model comparisons.

| Backend | Winner log loss delta, 95% interval | Podium log loss delta, 95% interval | Position MAE delta, 95% interval |
| --- | --- | --- | --- |
| CatBoost | -0.248 [-0.442, -0.059] | +0.011 [-0.153, +0.171] | -0.044 [-0.171, +0.086] |
| Histogram | -0.204 [-0.484, +0.170] | +0.053 [-0.118, +0.212] | +0.133 [-0.029, +0.297] |
| LightGBM | -0.067 [-0.372, +0.343] | +0.048 [-0.119, +0.206] | +0.109 [-0.040, +0.259] |
| XGBoost | +0.024 [-0.336, +0.476] | +0.058 [-0.108, +0.213] | +0.194 [+0.024, +0.387] |

The race-weighted podium Brier deltas versus logistic are -0.0088 for CatBoost,
+0.0033 for histogram boosting, +0.0033 for LightGBM and +0.0042 for XGBoost.
All four bootstrap intervals include zero. Pooled metrics weight driver rows;
these diagnostic deltas weight races equally, so their means differ slightly
from the protocol's pooled differences.

The largest podium log-loss failures are 2025 Great Britain, round 12, and
Sao Paulo, round 21. In Great Britain, histogram boosting assigned a 0.0000
rounded podium probability to Hulkenberg, who finished on the podium, while
assigning 0.9214 to Verstappen, who did not. In Sao Paulo, it assigned a 0.0000
rounded podium probability to Verstappen, who reached the podium, and 0.8074
to Leclerc, who did not. Each race contributes about +0.99 and +1.10 race-level
podium log-loss delta against logistic. The other backends have the same two
largest failures. Their errors are extreme enough to dominate the aggregate.

## Calibration, inputs and complexity

For histogram boosting, 61 podium predictions at 0.7 or above average 0.862,
while the observed podium rate is 0.689. LightGBM's corresponding 62 average
0.859 with observed 0.677; XGBoost's 58 average 0.867 with observed 0.672.
Low-probability podium bins underpredict some surprise finishes. Histogram
boosting's DNF predictions below 0.1 average 0.045 with observed rate 0.124
over 185 known rows. These are descriptive bins, not a fitted post-hoc
calibration correction.

Only qualifying position and last-session time have 583 of 594 values;
teammate qualifying delta has 574. Grid position and 15 other optional Core
values are entirely missing. Audited rolling finish means have 299, 219 and 60
values for windows 3, 5 and 10. Rolling DNF rates have 263, 184 and 48 values.
The corresponding missingness flags are available, but many are constant.
Driver and constructor IDs are context, not model features. The current model
matrix is numeric; no native categorical encoding is used. All four backends
receive training-only median imputation with empty columns retained, so many
declared predictors contribute no varying value in early folds. Logistic also
uses fold-fitted scaling. Boosting fits 80 iterations or trees with fixed small
depth or leaf limits. The first fit has 40 driver rows and 34 known DNF labels;
the last has 550 and 505. This is a plausible complexity and calibration limit,
but the audit does not identify a single causal mechanism.

Two controlled masks were run on identical folds and labels. Removing the
rolling fields changes mean winner log loss, full minus masked, by -0.236 for
histogram boosting, -0.019 for CatBoost, -0.016 for LightGBM and +0.073 for
XGBoost. The qualifying-only mask produces the same metrics as no-rolling
because the other Core numeric fields are empty or constant in this cohort.
No group has a uniform benefit across backends and tasks. These masks are
diagnostics only and are excluded from model selection; they did not alter the
frozen protocol.

Next work is to audit the two excluded final-outcome document chains, capture
future Gold races, and investigate podium tail behavior and calibration with
independent future races. The valid current decision is `no_selection`.
