# Historical Gold enrichment, 1 October 2026

## Frozen versions and evidence

The 95-race, 1,927-row post-qualifying Gold cohort remains fixed. Original Gold
SHA-256 is recorded in `data/benchmarks/gold_core/manifest.json`.
Rolling Gold SHA-256: `df1c47d8ed4dc1a7b55c85e9617633c205521e96f5d291ca5cb68d0ffb0d5ee6`.
The new historical enrichment is under
`data/benchmarks/gold_historical_enrichment_v2/<rolling-gold-sha>/<grid-audit-sha>/<practice-audit-sha>`.
Its Gold Parquet SHA-256 is
`442c5a0667c222555d1e4de18ad1673ad41bb1b2e02b3ba621633ea87694988f`.
Existing benchmark versions and prior experiment manifests were not modified.

The builder binds each prior final outcome to its Gold registry hash and checks
label publication before the target cutoff. Constructor identities and current
teammate membership come from the contemporary Gold roster, not from later team
names. The FIA grid audit hash is
`3cdd7d8fccf790605f5a35167dbf1f5d10c8a5aafc46c098b432b66f3ce8bf23`;
the FIA practice audit hash is
`8f223f425966646701f526cb5fa31e360d18153e2ec4a06a90cf14d13585c2e1`.
Their retained registry pages supply document identities and publication times;
the retained PDFs supply the exact values. The build rechecks hashes, timestamps,
roster mapping, and parsed values. Per-driver proof is in `feature_provenance.json`.

## Coverage

The original Gold has 1,905 qualifying positions. Every feature family below
started at zero coverage. Counts are nonmissing driver-race rows after enrichment.
All denominators are 1,927, except the annual denominators in the first column.

| Season (rows) | Constructor finish 3 / 5 / 10 | Constructor qualifying / teammate form | Grid | Practice rank / lap gap / teammate delta |
| --- | ---: | ---: | ---: | ---: |
| 2022 (320) | 120 / 40 / 0 | 40 / 40 | 0 | 319 / 316 / 312 |
| 2023 (360) | 120 / 80 / 0 | 80 / 79 | 0 | 359 / 353 / 348 |
| 2024 (459) | 340 / 280 / 180 | 280 / 277 | 0 | 459 / 449 / 438 |
| 2025 (480) | 420 / 380 / 280 | 380 / 379 | 360 | 460 / 458 / 456 |
| 2026 (308) | 176 / 110 / 0 | 110 / 109 | 132 | 307 / 304 / 300 |
| **Total (1,927)** | **1,176 / 890 / 460** | **890 / 884** | **492** | **1,904 / 1,880 / 1,854** |

`session_relative_rank` has the same coverage as practice rank. All six
championship fields, three constructor points windows, and constructor DNF rate
remain zero. Historical forecast air temperature, precipitation probability,
wind speed, and humidity remain zero. The machine-readable
`enrichment_report.json` records every added feature's missing reason by row and
coverage by season. It also lists all **71 races** whose FIA grid was published
after the existing post-qualifying cutoff. Those races retain missing grid values.

The main missing reasons are exact. Constructor finish 3 misses 265 rows for too
few prior rounds and 486 for an unaudited race in the window. Finish 5 misses
427 and 610; finish 10 misses 937 and 530. Qualifying form and teammate form use
the five-race window; teammate form additionally misses six rows with no
classified prior finish for a current teammate. Of 1,927 practice rows, 20 are
from one race whose latest FIA timing table cannot be conservatively parsed and
three drivers are absent from the selected practice classification. Printed lap
times are absent for another 24 drivers. Teammate lap delta has a further 50
rows without a current teammate's printed lap. All 1,927 championship and points
window values lack an audited scoring and sprint ledger. The current 95-race
cohort has no explicit binary DNF labels, so 890 otherwise eligible five-race
constructor windows still lack DNF rate. The 71 late-grid races account for all
1,435 missing grid rows.

## Why fields remain missing

Championship and constructor points are **not yet implemented in historical
Gold**, rather than inherently impossible to certify. FIA championship points
documents exist, but the retained final race ordinals do not encode sprint
awards, fastest-lap bonuses where applicable, reduced race points, or all
amendments. A complete per-event audited scoring ledger, with each used result's
availability bound and constructor membership, is required before deriving
standings, gaps, or points windows. Present-day standings responses are excluded.

The FIA grid registry supports publication-time checks. Only 24 race grid
documents were available by these cutoffs; no target race result was used to
backfill grid position. Later documents belong to a separately timed
provisional-grid cohort if that cohort is built and evaluated. The current
post-qualifying predictions and gates are unchanged.

FIA practice classifications do support conservative rank and printed-lap
features in 94 races. Retrospective FastF1/OpenF1 fetches were not automatically
promoted to Gold, and fuel-corrected race pace was not claimed. Open-Meteo's
[Single Runs API](https://open-meteo.com/en/docs/single-runs-api) exposes archived
model runs, but initialization is not release. Its
[model updates documentation](https://open-meteo.com/en/docs/model-updates)
describes live availability metadata; a historical per-run release certificate
for the target cutoffs was not established. Observed race weather is not a
forecast, so all weather values remain missing.

## Frozen evaluation and ablations

The enriched version was evaluated with the same 95-race cohort, chronological
folds, seed 42, 512 simulation draws, and CPU protocol as the prior 95-race
Gold run. Each backend has 92 paired held-out races. Relative to the previous
version, race-paired bootstrap intervals (new minus old loss, 95%) were:

| Backend | Winner log loss | Podium Brier | Position MAE |
| --- | ---: | ---: | ---: |
| CatBoost | -0.009 [-0.160, 0.106] | +0.001 [-0.001, 0.003] | +0.001 [-0.034, 0.034] |
| HistGradientBoosting | -0.286 [-0.976, 0.106] | +0.001 [-0.001, 0.003] | +0.008 [-0.024, 0.042] |

Single-group ablations compared constructor, grid, and practice features to a
masked rolling baseline on the same enriched schema. Every paired 95% interval
for each group's winner log loss, podium Brier, and position MAE included zero.
The largest numeric winner gain was practice with HistGradientBoosting, -0.297,
but its interval was [-0.987, 0.093]. The analogous CatBoost practice delta was
+0.353 [-0.083, 1.045]. These shifts are unstable and do not justify selection.
Championship, constructor points, DNF rate, and weather cannot be ablated as
available features because their Gold coverage is zero. The frozen gate retains
`no_selection` / deferred; no validated forecast gate was changed. DNF metrics
have no known labels on this 95-race cohort.

The local comparison artifacts are under
`models/experiments/gold/enrichment_comparison/442c5a0667c222555d1e4de18ad1673ad41bb1b2e02b3ba621633ea87694988f`
and `models/experiments/gold/enrichment_ablations/442c5a0667c222555d1e4de18ad1673ad41bb1b2e02b3ba621633ea87694988f`.
These and the immutable benchmark and exact FIA audit bytes are ignored local
artifacts; the scripts reproduce them from retained source evidence.
