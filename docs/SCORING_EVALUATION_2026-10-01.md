# Audited championship scoring evaluation, 2026-10-01

The committed FIA audit covers 2022 to 2026. The native importer verified retained FIA points documents, regulation sources, and the collection index, then reconstructed 117 publication-time versions for 107 completed events. The scoring ledger hash is `ca68557d139052b3ae928838c0479bf618ed61d86563132ea8dbd4863d5f0a53`. The new immutable [Gold scoring benchmark](../data/benchmarks/gold_championship_scoring_v2/ecbe5241aa69c2207c454775614883995600ce1f10a1023ab2c4775a8c05c277/ca68557d139052b3ae928838c0479bf618ed61d86563132ea8dbd4863d5f0a53/manifest.json) has 95 races and 1,927 driver-race rows. It preserves the previous benchmark and model runs.

## Point-in-time coverage

Only a complete, undisputed earlier event published before the prediction cutoff contributes points. The ledger does not infer points from finish positions or today's standings. The [coverage and missingness report](../data/benchmarks/gold_championship_scoring_v2/ecbe5241aa69c2207c454775614883995600ce1f10a1023ab2c4775a8c05c277/ca68557d139052b3ae928838c0479bf618ed61d86563132ea8dbd4863d5f0a53/scoring_report.json) records each field by season.

| Season | Gold races | Rows | Driver and constructor points | Driver position | Constructor position | Last 3 driver points | Last 5 | Last 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 16 | 320 | 260 | 240 | 240 | 240 | 240 | 140 |
| 2023 | 18 | 360 | 260 | 240 | 240 | 240 | 200 | 140 |
| 2024 | 23 | 459 | 439 | 419 | 419 | 380 | 340 | 260 |
| 2025 | 24 | 480 | 420 | 400 | 400 | 400 | 380 | 280 |
| 2026 | 14 | 308 | 242 | 218 | 220 | 198 | 176 | 66 |
| **Total** | **95** | **1,927** | **1,621** | **1,517** | **1,519** | **1,458** | **1,336** | **886** |

The 306 rows without prior points have `audited_prior_event_missing_or_uncertain`. Positions can be missing despite known points when the audited race placing countback does not resolve a tie. Rolling windows require every preceding round and remain missing early in a season. Per-row [scoring provenance](../data/benchmarks/gold_championship_scoring_v2/ecbe5241aa69c2207c454775614883995600ce1f10a1023ab2c4775a8c05c277/ca68557d139052b3ae928838c0479bf618ed61d86563132ea8dbd4863d5f0a53/scoring_provenance.json) binds the exact event version, publication clock, awarded components, source references, and evidence hashes.

## Frozen evaluation and ablations

The prior version and scored version were run on the same 92 paired outer test races with seed 42, 4,096 draws, CPU, and the frozen chronological protocol. Values below are from the [paired version comparison](../models/experiments/gold/scoring_ablations/e7579a124f444474f8d66dc83dd6a35488377229d46708a18e74e63e3d6c0c59/version_comparison.json). Lower is better for every metric. The 95% intervals resample whole races and describe scored minus prior loss.

| Backend | Winner log loss, prior to scored | Paired winner delta, 95% interval | Podium Brier, prior to scored | Position MAE, prior to scored |
| --- | ---: | ---: | ---: | ---: |
| Hist | 1.4670 to 1.4612 | -0.0058 [-0.0707, 0.0602] | 0.0815 to 0.0817 | 3.0978 to 3.0928 |
| CatBoost | 1.4716 to 1.4021 | -0.0694 [-0.1933, 0.0300] | 0.0811 to 0.0797 | 3.0476 to 3.0152 |

The [scoring ablations](../models/experiments/gold/scoring_ablations/e7579a124f444474f8d66dc83dd6a35488377229d46708a18e74e63e3d6c0c59/summary.json) mask driver points, constructor points, or all championship points independently. They use the same folds, seed, draws, and race-level bootstrap. Deltas below are ablated minus full model. A negative value means the feature group worsened that loss in this run.

| Removed features | Backend | Winner log loss delta, 95% interval | Podium Brier delta, 95% interval | Position MAE delta, 95% interval |
| --- | --- | ---: | ---: | ---: |
| Driver points | Hist | -0.0476 [-0.0993, -0.0007] | -0.00015 [-0.00101, 0.00068] | 0.0045 [-0.0152, 0.0228] |
| Driver points | CatBoost | 0.0146 [-0.0441, 0.0729] | 0.00065 [-0.00026, 0.00166] | 0.0107 [-0.0037, 0.0255] |
| Constructor points | Hist | -0.0126 [-0.0649, 0.0490] | -0.00014 [-0.00112, 0.00089] | -0.0215 [-0.0430, -0.0007] |
| Constructor points | CatBoost | 0.0005 [-0.0545, 0.0557] | 0.00077 [-0.00034, 0.00201] | 0.0081 [-0.0121, 0.0262] |
| All points | Hist | 0.0058 [-0.0602, 0.0707] | -0.00025 [-0.00213, 0.00160] | 0.0070 [-0.0307, 0.0475] |
| All points | CatBoost | 0.0694 [-0.0300, 0.1933] | 0.00135 [-0.00028, 0.00300] | 0.0321 [-0.0060, 0.0761] |

The combined points feature group has no clear paired improvement: all three whole-group intervals cross zero in both backends. Individual exploratory intervals show a Hist winner regression with driver points and a Hist position regression with constructor points. These are multiple diagnostics, so they do not establish a stable feature selection decision. The [full model report](../models/experiments/gold/dataset-a4ee51c6aea4a7703b44616e6cc15dcc5506efeea60ec0e17a9ca4a1640f8c85/runs/395bc6cc4ac841f78c605cede0f490b0/comparison.json) remains `deferred` for all four backends under the existing baseline and uncertainty gates. There is no selected model or validated WDC or WCC forecast from this milestone.

Constructor rolling DNF features remain on the separate audited Gold outcome path. Their sparse binary labels cannot fill scoring gaps and did not affect this points evaluation.
