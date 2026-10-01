# Audited championship scoring evaluation, 2026-10-01

The committed FIA audit covers 2022 to 2026. The native importer verified retained FIA points documents, regulation sources, and the collection index, then reconstructed 117 publication-time versions for 107 completed events. The scoring ledger hash is `6730fc5f8a60a9b2ff83ae8a593b19d99c7320890132f34f8a076ecdeded8a79`. The new immutable [Gold scoring benchmark](../data/benchmarks/gold_championship_scoring_v3/ecbe5241aa69c2207c454775614883995600ce1f10a1023ab2c4775a8c05c277/6730fc5f8a60a9b2ff83ae8a593b19d99c7320890132f34f8a076ecdeded8a79/manifest.json) has 95 races and 1,927 driver-race rows. It preserves the previous benchmark and model runs.

## Point-in-time coverage

Only a complete, undisputed earlier event published before the prediction cutoff contributes points. The ledger does not infer points from finish positions or today's standings. The [coverage and missingness report](../data/benchmarks/gold_championship_scoring_v3/ecbe5241aa69c2207c454775614883995600ce1f10a1023ab2c4775a8c05c277/6730fc5f8a60a9b2ff83ae8a593b19d99c7320890132f34f8a076ecdeded8a79/scoring_report.json) records each field by season.

| Season | Gold races | Rows | Driver and constructor points | Driver position | Constructor position | Last 3 driver points | Last 5 | Last 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 16 | 320 | 260 | 236 | 240 | 240 | 240 | 140 |
| 2023 | 18 | 360 | 260 | 238 | 240 | 240 | 200 | 140 |
| 2024 | 23 | 459 | 439 | 400 | 419 | 380 | 340 | 260 |
| 2025 | 24 | 480 | 420 | 396 | 400 | 400 | 380 | 280 |
| 2026 | 14 | 308 | 242 | 212 | 216 | 198 | 176 | 66 |
| **Total** | **95** | **1,927** | **1,621** | **1,482** | **1,515** | **1,458** | **1,336** | **886** |

The 306 rows without prior points have `audited_prior_event_missing_or_uncertain`. Positions can be missing despite known points when the audited race placing countback does not resolve a tie. Four 2022 United States race placings differ between the FIA points matrix and audited classification; affected tied positions stay missing. Rolling windows require every preceding round and remain missing early in a season. Per-row [scoring provenance](../data/benchmarks/gold_championship_scoring_v3/ecbe5241aa69c2207c454775614883995600ce1f10a1023ab2c4775a8c05c277/6730fc5f8a60a9b2ff83ae8a593b19d99c7320890132f34f8a076ecdeded8a79/scoring_provenance.json) binds the exact event version, publication clock, awarded components, source references, and evidence hashes.

## Frozen evaluation and ablations

The prior version and scored version were run on the same 92 paired outer test races with seed 42, 4,096 draws, CPU, and the frozen chronological protocol. Values below are from the [paired version comparison](../models/experiments/gold/scoring_ablations/644db66cf35f23866cd75cdf474bd4c0bb54d3e74bb3e426dfcd63041b794259/version_comparison.json). Lower is better for every metric. The 95% intervals resample whole races and describe scored minus prior loss.

| Backend | Winner log loss, prior to scored | Paired winner delta, 95% interval | Podium Brier, prior to scored | Position MAE, prior to scored |
| --- | ---: | ---: | ---: | ---: |
| Hist | 1.4670 to 1.4561 | -0.0110 [-0.0756, 0.0536] | 0.0815 to 0.0817 | 3.0978 to 3.0935 |
| CatBoost | 1.4716 to 1.4044 | -0.0672 [-0.1890, 0.0299] | 0.0811 to 0.0797 | 3.0476 to 3.0157 |

The [scoring ablations](../models/experiments/gold/scoring_ablations/644db66cf35f23866cd75cdf474bd4c0bb54d3e74bb3e426dfcd63041b794259/summary.json) mask driver points, constructor points, or all championship points independently. They use the same folds, seed, draws, and race-level bootstrap. Deltas below are ablated minus full model. A negative value means the feature group worsened that loss in this run.

| Removed features | Backend | Winner log loss delta, 95% interval | Podium Brier delta, 95% interval | Position MAE delta, 95% interval |
| --- | --- | ---: | ---: | ---: |
| Driver points | Hist | -0.0425 [-0.0944, 0.0034] | -0.00016 [-0.00098, 0.00062] | 0.0034 [-0.0150, 0.0207] |
| Driver points | CatBoost | 0.0123 [-0.0462, 0.0708] | 0.00072 [-0.00023, 0.00175] | 0.0103 [-0.0045, 0.0248] |
| Constructor points | Hist | -0.0147 [-0.0731, 0.0526] | -0.00013 [-0.00112, 0.00086] | -0.0204 [-0.0408, 0.0001] |
| Constructor points | CatBoost | 0.0002 [-0.0527, 0.0552] | 0.00083 [-0.00031, 0.00204] | 0.0050 [-0.0159, 0.0229] |
| All points | Hist | 0.0110 [-0.0536, 0.0756] | -0.00026 [-0.00206, 0.00157] | 0.0059 [-0.0305, 0.0454] |
| All points | CatBoost | 0.0672 [-0.0299, 0.1890] | 0.00141 [-0.00025, 0.00310] | 0.0317 [-0.0060, 0.0739] |

The combined points feature group has no clear paired improvement: all whole-group intervals cross zero in both backends. Individual driver and constructor ablations also have intervals spanning zero. The [full model report](../models/experiments/gold/dataset-b3324b64c0aa6ae41ee8239ef71859ad6a188295232a3d755a35d1702b2bae9a/runs/83155b99f0cf4944b653b1c70829990e/comparison.json) remains `deferred` for all four backends under the existing baseline and uncertainty gates. There is no selected model or validated WDC or WCC forecast from this milestone.

Constructor rolling DNF features remain on the separate audited Gold outcome path. Their sparse binary labels cannot fill scoring gaps and did not affect this points evaluation.
