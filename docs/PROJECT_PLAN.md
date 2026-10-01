# F1 ML Predictor Project Plan

Status: Phases 1 through 7 and Phase 9 simulation infrastructure are delivered. Historical Gold now contains 95 post-qualifying races and 1,927 driver-race observations. The audited championship scoring benchmark has 1,621 rows with point-in-time driver and constructor totals; its frozen evaluation has 92 paired outer test races. The separate audited binary DNF benchmark retains 29 races, with 547 known labels and 26 paired test races. Task-specific model selection remains `no_selection` because of baseline regressions or uncertain paired improvement. Historical Gold eligibility remains open to evidence-based upgrades; each experiment binds an immutable benchmark version. A historical held-out race has a `development_only` prediction artifact, and `predict-next-race` now publishes `development_only` next-race and season predictions. Validated race and championship forecasts still require prospective independent confirmation. The Windows prospective task remains disabled pending a local credential for signed-out network access; manual collection remains available. Updated 2026-10-01.

## Development prediction pipeline milestone

`predict-next-race` finds the next race from a fresh collector schedule
observation and prefers a certified post-qualifying capture. Before qualifying,
it publishes a clearly labelled `pre_qualifying` run at the run time instead.
It freezes a content-addressed snapshot built with the Gold rolling,
constructor-form and scoring-ledger rules from audited history published before
the cutoff. It fits development candidates on the latest immutable Gold
versions because every task remains `no_selection`. The position model is the
joint backend with the best mean rank on winner, podium and finish losses in the
frozen comparison. DNF comes from the separate audited binary DNF version, so it
is independent of pace. Predictors unavailable at the cutoff are hidden from
training as well as prediction. A cached cutoff-matched diagnostic evaluation
and the existing baselines are reported alongside. The existing joint sampler
and championship simulator supply race distributions and development WDC/WCC
probabilities, expected points and final position distributions. The run fails
closed unless coherence, cutoff, immutable dataset, sample provenance and
`development_only` labelling checks pass. See
[development predictions](DEVELOPMENT_PREDICTIONS.md). The first run covers 2026
round 16 before qualifying. None of these outputs is a validated forecast.

## Audited championship scoring milestone

The season and round aware scoring ledger importer, point-in-time standings engine,
and version 4 Gold scoring benchmark are implemented using the committed FIA audit.
Retained source hashes and the native evidence schema are verified before use. The
new benchmark preserves the prior version and binds each snapshot to a scoring
ledger hash and publication-time event versions. Coverage, the frozen chronological
evaluation, and separate driver and constructor point ablations are in the
[scoring evaluation](SCORING_EVALUATION_2026-10-01.md). Paired evidence does not
justify selecting a backend; the existing `no_selection` gate remains in force.
See [scoring ledger handoff](SCORING_LEDGER.md) for the schema and build command.
Constructor rolling DNF evidence remains a separate sparse, audited feature path.

## Evolving Gold and reproducible model runs

The Gold registry is not permanently frozen. Before and after a benchmark rebuild,
the writer verifies and copies the exact manifest, coverage report and tier Parquet
bytes into `data/benchmarks/<benchmark>/versions/dataset-<full manifest SHA-256>/`.
Each version has a race-level `race_status.json` with prior status and evidence
references. An `excluded_to_gold` transition is explicit when an excluded race gains
Gold evidence. The mutable `versions/current.json` pointer identifies the latest
snapshot; earlier versions and their eligibility decisions stay intact. A new
version is staged and verified before its directory is published, so an
interrupted copy cannot expose a partial benchmark as a valid model input.

Every CLI baseline or probabilistic comparison reads a verified snapshot and writes
to a unique `runs/<run_id>` directory. The report records dataset version and
manifest hash, Gold and driver-race counts, feature and evaluation protocol versions,
model settings, per-fold fit/calibration cutoffs, random seed, dependency versions,
actual CPU/GPU device, metrics and creation time. Old run files are never reused.
The original 95-race rolling benchmark remains the input for the current comparison;
future Gold upgrades create a new version and a new run. Historical source and
prediction hashes remain in the run report. A newly selected version must be frozen
before prospective testing; future outcomes cannot tune that version.

The current 95-race version has manifest SHA-256
`6d854400f69fc0c0724626d26544124dc4705b6653e9b1377670c41acedebd94`.
Its [version-2 comparison](../models/experiments/gold/dataset-6d854400f69fc0c0724626d26544124dc4705b6653e9b1377670c41acedebd94/runs/a0fb69d0ee4c47199b4405551484f79d/comparison.json)
has 92 paired races. Logistic has the lowest observed winner log loss (1.363),
podium log loss (0.250) and finish MAE (2.713). CatBoost, the strongest boosting
backend on those tasks, records 1.445, 0.253 and 3.045 respectively. The 29-race
[audited DNF comparison](../models/experiments/gold/dataset-be0704f0618d608743ce3bf89adf033305a52b43b421d6d6fdcefaa912caae0f/runs/6216cd9ee42d452993c1d80e6f6f7249/comparison.json)
has 26 paired races and 494 known DNF test labels. Logistic has the lowest DNF
Brier score (0.1114); differences from boosting are small and paired intervals
overlap zero. CatBoost has the lowest observed winner loss on that smaller version,
but a top-1 baseline regression prevents provisional winner selection. The separate
versions must not be pooled into one reported test score.

Seven cumulative [feature ablations](../models/experiments/gold/ablations/df1c47d8ed4dc1a7b55c85e9617633c205521e96f5d291ca5cb68d0ffb0d5ee6/summary.json)
use the same 92 outer races and preserve missingness. Qualifying position is present
for 1,905 of 1,927 rows; grid, constructor, championship, practice and weather
predictors are absent throughout this version. Rolling finish windows are present
for 1,163, 884 and 457 rows at lengths 3, 5 and 10. The recent-form group improves
CatBoost rank MAE versus qualifying only by about 0.05 with a race-bootstrap 95%
interval of about 0.02 to 0.09 lower error, while its winner-loss change remains
uncertain. Teammate features give a small HistGradientBoosting rank improvement.
Groups with no coverage yield identical predictions. These are exploratory feature
diagnostics, not a protocol change selected on final held-out races.

The [95-race reliability audit](../models/experiments/gold/dataset-6d854400f69fc0c0724626d26544124dc4705b6653e9b1377670c41acedebd94/runs/a0fb69d0ee4c47199b4405551484f79d/calibration_audit_v2.json)
keeps podium tail failures visible. The [paired DNF calibration audit](../models/experiments/gold/calibration/sigmoid-vs-isotonic-29.json)
tests sigmoid and isotonic with six earlier calibration races and identical outer
folds. Isotonic fits fewer folds; its Brier differences have intervals spanning
zero, so there is no calibration switch. The existing Gumbel/Plackett-Luce joint
sampler supplies winner, podium and finishing marginals from complete orders while
DNF is a separate probability. A published [development-only held-out prediction](../data/predictions/development/dataset-be0704f0618d608743ce3bf89adf033305a52b43b421d6d6fdcefaa912caae0f/6216cd9ee42d452993c1d80e6f6f7249/catboost.parquet)
for 2026 round 15 has no outcome columns and passes probability coherence and
roster checks. It is retrospective and does not count as prospective validation.
CatBoost, LightGBM and XGBoost ranking objectives remain candidates for a future
version only if they beat the current rank baseline on the same chronological
race groups without using later outcomes.

## Product and prediction contract

The first supported prediction snapshot is **after qualifying and before the race**. A prediction is keyed by season, round, and a UTC `prediction_timestamp`. Every input must have an auditable `available_at` timestamp no later than that cutoff. The initial product returns winner, podium, finishing position, and DNF estimates for each entered driver. Championship probabilities, explanations, and a dashboard follow after race-level predictions are backtested.

Success means a user can reproduce a historical prediction using only information available at its original cutoff. Accuracy claims must come from chronological rolling backtests, not a random row split.

## Architecture and data contracts

- `src/f1_ml_predictor/`: installable Python package. Keep `sources`, `ingestion`, `normalization`, `features`, `models`, `simulation`, and `explain` separate as those phases arrive. A future `api` or dashboard package reads published predictions and never owns ingestion or training logic.
- `data/raw/<source>/`: versioned unnormalized response collections with request endpoint, first retrieval time, source publication time when supplied, and content hash. Cache unchanged historical responses. Do not put large telemetry archives here by default.
- `data/normalized/`: typed Parquet tables for events, entries, qualifying, race results, sessions, weather snapshots, and provenance. Preserve source identifiers and raw references.
- `data/features/`: Parquet feature snapshots keyed by event, driver, cutoff, and feature version. Every feature records or derives an availability bound.
- `models/`: fitted model files and manifests with training window, features, code version, and metrics.
- `data/predictions/`: Parquet outputs with model version, cutoff, probabilities, ranks/distributions, and provenance. All data and model artifact directories stay out of git.
- Stable identifiers: Jolpica `driverId`, `constructorId`, and `circuitId` are canonical where available. An event is `(season, round)`; a session adds a session code. Store aliases and source IDs in explicit crosswalks. Never join on display names or current driver numbers.
- Times are timezone-aware UTC. `available_at` means the earliest defensible time this specific value could be known, not merely when it was fetched. Unknown historical publication time is represented as unknown and excluded from strict backtests until resolved. Version corrected records rather than rewriting a historical snapshot.
- Ingestion is partitioned by source, season, event, and session as appropriate. Maintain resumable manifests and atomic writes. Retry bounded transient failures with timeouts and source-specific throttling. Normal test runs use recorded small fixtures or mocks, never live APIs.
- Parquet is the first tabular storage format. Avoid a database, distributed scheduler, large telemetry downloads, and neural networks until measured needs justify them.

## Source responsibilities and verified documentation

| Source | Initial responsibility | Integration constraints and leakage notes |
| --- | --- | --- |
| [Jolpica](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md) | Schedules, stable IDs, entries, results, qualifying, points, standings, pit stops | Historical backbone. Paginate explicitly, identify the client, obey [documented limits](https://github.com/jolpica/jolpica-f1/blob/main/docs/rate_limits.md) of 4 requests/second and 500/hour for unauthenticated use. Current standings and corrected results cannot be used as historical pre-race inputs without an as-of record. |
| [FastF1](https://github.com/theOehrly/Fast-F1) | Practice and qualifying lap, tyre, and selected pace summaries | Use its [cache and rate-limit behavior](https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/cache_and_rate_limits.rst). Defer detailed telemetry. Session data must be available by the cutoff. |
| [OpenF1](https://openf1.org/docs/) | Recent session timing, stints, pits, grid, weather, and cross-checks | Primarily 2023 onward. Free historical API is documented at 3 requests/second and 30/minute; live access requires paid authentication. Same-race race data cannot enter pre-race features. |
| [Open-Meteo](https://open-meteo.com/en/docs/single-runs-api) | Captured forecasts and exact archived model runs | Model initialization is not public release. Require evidence for archived availability; do not use stitched historical forecasts or actual weather as as-of forecasts. Check [plan limits and terms](https://open-meteo.com/en/pricing) before scheduled use. |
| [FIA](https://www.fia.com/documents) | Authoritative classification, grid, penalty, and rule audit | Public documents have publication times and provisional/final versions. No general public JSON API was established; begin with manual audit links, then review rights and automation feasibility. |

Before each client is implemented, recheck its official documentation for endpoints, authentication, limits, and terms. Packaging follows the [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/); Parquet writing will follow [PyArrow documentation](https://arrow.apache.org/docs/python/parquet.html).

## Leakage controls and evaluation

- Enforce `feature_timestamp <= prediction_timestamp` for every input. Also track source publication time and ingestion time; a late retrieval is not proof of historical availability.
- Qualifying results, grid, and penalties are usable only after publication at the chosen cutoff. If an official grid was not yet known, use the latest known grid and label its status.
- Lag form, reliability, constructor pace, pit performance, points, and standings to completed prior events. Use contemporaneous constructor membership. Never use final season standings, future rounds, post-race corrections, unknown strategy, race stints, or actual race weather as pre-race features.
- Historical forecasts must represent a run issued before cutoff. Missing run history stays missing, with a missingness indicator, rather than being filled from observed race weather.
- Split by event chronology, with multiple rolling origins and season boundaries. Keep all drivers from an event in the same fold. Report log loss, Brier score, calibration, ranking measures, top-N accuracy, position error, and DNF metrics. Compare against transparent heuristics and logistic regression before adding complexity.
- Winner probabilities must sum to one per event. Podium probabilities and finish distributions must be coherent with field size; report calibration and uncertainty, including for rare DNFs.

## Phases

Each phase ends with focused tests, formatting/lint/type checks where configured, diff and artifact review, plan update, one clean commit, commit inspection, and a push to the configured remote. A failed push leaves the local commit intact and is reported with its exact error.

Commit subjects describe the delivered engineering capability. Commit bodies explain scope, rationale, and verification evidence so the history is useful to collaborators and recruiters. Keep claims factual and do not rewrite published history without explicit instruction.

| Phase | Objective and dependencies | Tests and acceptance criteria | Main risk and output |
| --- | --- | --- | --- |
| 1. Foundation | No upstream dependency. Create package metadata, storage zone paths, stable event/entity identifiers, UTC cutoff contract, test conventions, and developer setup. | Clean package import; deterministic unit tests for IDs, paths, and cutoff rejection; no live API or generated data required; complete plan exists before code. | Risk: premature abstractions. Output: installable skeleton and enforceable base contracts. |
| 2. Historical backbone | Depends on 1. Add Jolpica client, retry/timeout/throttle/pagination, incremental raw cache/manifests, normalized event/entry/qualifying/result tables in Parquet. | Mocked API and schema tests cover pagination, 429/5xx, malformed and missing data, resumability, no duplicate partitions, stable IDs; a bounded real smoke fetch is optional. | Risk: retrospective data revisions. Output: reproducible historical core dataset with provenance. |
| 3. Session and forecast enrichment | Depends on 2. Add bounded FastF1/OpenF1 summaries and Open-Meteo forecast snapshots; FIA audit metadata where available. | Mocked clients and integration tests prove session cutoff, forecast run cutoff, missing-data behavior, cache reuse, and source mapping; no bulk telemetry. | Risk: source coverage and weather leakage. Output: versioned enrichment tables. |
| 4. Feature snapshots | Depends on 2, optionally 3 for enriched columns. Build lagged form, qualifying/grid, teammate, constructor, reliability, circuit, points, weather, practice, tyres, and pit features with explicit missingness. | Feature and leakage tests inspect every field's maximum availability time; snapshots rebuild deterministically; no future result changes an earlier snapshot. | Risk: silent retrospective joins. Output: typed feature dataset and feature dictionary. |
| 5. Data trust and as-of certification | Extend 4, do not rebuild it. Add explicit evidence classes, tier policy, per-feature provenance, named cutoffs, immutable prospective collection, audited outcomes, provider arbitration, cancelled qualifying and pit-lane schemas. | Offline tests reject quality upgrades, backdated live captures, late revisions, ambiguous provider data, weather-init-as-release, invalid start states, and unsafe snapshot overwrite. Legacy features stay usable but Development-labelled. | Risk: confusing publication claims with verified value versions. Output: certification contracts, audit metadata, capture CLI, conservative feature vocabulary. |
| 6. Evidence-tiered benchmarks | Depends on 5. Build Gold, Silver, and Development datasets with final audited outcome labels, machine-readable manifests, and race/feature coverage and exclusion reports. | Deterministic partitions and hashes; event-complete label joins; no label columns used as features; no tier promotion; empty Gold explicitly reported when unsupported. | Risk: sparse certified coverage and exploratory leakage. Output: three Parquet datasets, benchmark manifest, coverage report. |
| 7. Baselines and chronological backtests | Depends on 6. Heuristic and logistic winner/podium/DNF baselines, position baseline, rolling event-grouped evaluation, calibration reports and manifests. | Training-only transforms, time-ordered folds, label availability at training time, deterministic seeds, model smoke tests, probability coherence, per-tier metrics and honest insufficient-data states. | Risk: small sample and imbalance. Output: baseline/evaluation infrastructure; primary accuracy claims only on Gold. |
| 8. Stronger probabilistic race models | Software delivered. Retain existing adapters, device policy, grouped folds and paired baselines. Expand audited historical Gold Core and capture future races. | At least 15 distinct paired Gold test races for preliminary comparison and 25 for provisional selection, with no baseline regressions and all task metrics evaluated. Calibration remains earlier-history only. | Risk: sparse direct evidence. Output: reproducible comparisons; selected model remains data-gated. |
| 9. Championship simulation and explanations | Simulation infrastructure delivered using the shared joint race sampler and explicit synthetic inputs. Validated real outputs and model explanations depend on Phase 8 Gold acceptance. | Whole-order sampling, fixed seeds, season/event scoring, explicit classification eligibility, constructor transfers, unresolved countback ties, serialization and provenance. | Risk: changing rules and compounded uncertainty. Output: engineering simulation infrastructure now; real probabilities and compatible explanations later. |
| 10. Dashboard and operations | Depends on 7 for initial interface and 9 for full scope. Separate read-only API/UI for published predictions, evidence tier, uncertainty and freshness; scheduled bounded refresh. | API/UI integration, responsive workflows, stale/missing states, deployment/runbook, offline suite. | Risk: presenting exploratory or stale outputs as reliable. Output: dashboard and operational collection workflow. |

## Remaining roadmap policy

Run two data tracks concurrently. Expand the historical candidate pool by direct
publication/version evidence and audit strongest excluded candidates first. Gold
eligibility may rise beyond the current 95-race version. Gold Core can
omit unverifiable optional inputs. Gold Full can grow through richer prospective
captures and is not a prerequisite for baseline evaluation. Expand the initial
eight Core races for the post-qualifying cohort without relaxing evidence
requirements. Retain per-snapshot manifests, availability matrices, hashes,
registry entries and explicit exclusion reasons. See [Gold workflow](GOLD_WORKFLOW.md).

Prospective ticks discover the next race and freeze fresh post-qualifying input
versions while historical audits proceed. Later audited outcomes attach separately.
Missed captures are never backdated. Phase 9 may use synthetic joint orders while
real model selection is blocked; fixture metrics and WDC/WCC outputs are engineering
checks only. Avoid exhaustive model searches before Gold coverage exists.

Phase 5 evidence classes are `captured_live`, `source_published_timestamp`, `versioned_archive`, `conservative_reconstruction`, and `current_state_only`. Evidence must bind to the exact value version. Captured-live availability equals capture time; published/archive evidence requires independently audited timestamp/version metadata; conservative reconstruction requires an explicit method and upper availability bound. Existing free-form references default to current-state/Development, never upgraded implicitly.

Gold requires verified direct pre-cutoff evidence for every used sensitive input. Silver permits audited conservative reconstruction with documented assumptions. Development contains legacy/current-state/unaudited exploratory inputs and cannot support primary accuracy claims. Missing optional features remain missing and visible in coverage. Final audited outcomes are labels, not predictive inputs; their later publication is allowed for evaluation but training folds must respect label availability.

Prospective collection freezes exact fresh API payloads and derived input versions with UTC capture times and hashes. Named post-qualifying, provisional-grid, and pre-race windows are separate prediction cohorts. A frozen cutoff cannot be overwritten or retrospectively populated. FIA audit metadata retains exact document identity, printed publication time/timezone, revision/recall status, and hash. Weather initialization is separate from actual availability. FastF1/OpenF1 arbitration selects a documented provider per session and flags disagreement, never averages incompatible filters.

Phase 6 runs against available local data without manufacturing Gold/Silver races. It emits valid empty tiers and explicit exclusion reasons when evidence or audited labels are absent. Baseline software can be developed on deterministic fixtures and clearly labelled Development data while certified coverage accumulates; fixture metrics are not real-world accuracy.

Benchmark catalogs bind exact feature and final audited outcome Parquet bytes to event, prediction timestamp, and named cutoff. A race/window joins only when its complete feature roster exactly matches audited labels, labels are published after the prediction cutoff, and exactly one winner is present. Outcome columns are prefixed `label_`; they never enter the feature column list. Multiple cutoff cohorts for one event are permitted. Each dataset has a content hash; the machine-readable manifest binds tier files and coverage.

## Phase 1 implementation gate

Acceptance requires: `pyproject.toml` defines an installable `src` package and test configuration; package code resolves all five storage zones without creating data on import; event and entity IDs reject malformed values; cutoff validation rejects naive timestamps and any source availability after the prediction cutoff; tests pass offline; ignored artifact paths include all planned zones; README gives concise local setup and current scope. No API download is part of Phase 1.

Phase 1 verification: editable install succeeded in a local virtual environment; 17 offline tests passed; Ruff lint and format checks passed; strict mypy passed; no forbidden character was found in changed content.

## Phase 2 result

Jolpica ingestion now covers schedules and per-event qualifying/results, with derived entries. The HTTP client paginates, identifies itself, enforces local request budgets, and retries bounded transient failures with timeouts. Raw source collections are content-addressed and versioned; normalized Parquet partitions are replaced atomically and skipped when their source hashes and schema version are unchanged. Failed jobs can resume from cached collections.

Historical seasons use the cache by default; current/future seasons refresh on each run. `--refresh` explicitly rechecks older data. Run one ingestion process per shared API budget. Normalized `available_at` remains null because Jolpica does not supply historical publication timestamps. Grid values in race results are outcomes for auditing, not pre-race features. DNF labels are deferred because status semantics vary and can be revised.

Verification: 54 offline tests passed, including mocked HTTP-to-Parquet integration and interrupted-run recovery. A bounded live 2025 smoke run completed 24 events; a repeat made zero API fetches, hit 49 cached collections, and skipped 73 unchanged Parquet partitions. Lint, format, type, and punctuation checks passed.

## Phase 3 result

OpenF1 discovery and session ingestion now cover completed practice and qualifying lap/stint/pit summaries. Driver numbers are mapped within the selected event to canonical IDs; conflicting aliases and mixed-session responses are rejected. Unmapped laps are reported rather than assigned invented identities. Summaries exclude pit laps and invalid timing; missing stint ranges retain missing tyre values. The optional FastF1 adapter loads lap timing and race-control messages needed for deleted-lap filtering, with telemetry and actual-weather loading disabled.

Open-Meteo live forecasts retain first capture-time availability. Exact archived runs require a model initialization time, independently supported release time, evidence reference, and prediction cutoff. Forecast target time and availability are separate columns. Validation rejects unsupported backdating, malformed arrays, nonfinite values, and non-UTC responses before cache persistence. FIA integration records official document links and supplied publication times for audit, not parsed grids or automated document scraping.

Enrichment responses and typed Parquet outputs are content-addressed. Unchanged captures preserve first retrieval time; changed inputs produce separate summary versions. Source identity, event crosswalks, and their retrieval timestamps participate in OpenF1 summary provenance. HTTP requests have bounded retries/timeouts and local source-specific budgets. Run one process per shared source budget. FastF1 remains an optional dependency; Windows installs `tzdata` for Parquet timezone conversion.

Verification: 112 offline tests passed, including mocked APIs, schema/integration checks, cache reuse, race/future-session rejection, mixed-session rejection, invalid tyre ages, and forecast backdating rejection. Live Australian 2025 qualifying smoke runs produced 21 OpenF1 and 22 FastF1 driver/compound rows; both reused unchanged outputs on rerun. The different counts reflect source-specific quality filters, not interchangeable pace estimates. Ruff lint/format and strict mypy passed. No bulk telemetry was downloaded.

Coverage limitations remain explicit: OpenF1 historical coverage begins in 2023. Open-Meteo Single Runs documentation lists IFS HRES 9 km coverage from 2024-03-14 and other listed models from 2026-04-02; verify model-specific coverage when requesting a run. Neither a session end timestamp nor a forecast initialization timestamp proves when corrected data was publicly available. Present-day historical fetches cannot establish strict old pre-race availability.

## Phase 4 implementation gate

Build deterministic feature snapshots over explicitly versioned, availability-stamped inputs. Require a published qualifying snapshot, entered-driver roster, race start, and prediction cutoff before the race. Reject unknown or late required inputs; leave unavailable optional inputs missing. Select prior-event result versions by availability, lag rolling form and reliability, and never derive the current event grid from race results. Weather targets may be in the future, but the forecast itself must be known by the cutoff. Persist a feature dictionary and per-row provenance bound. Tests must prove that future results, late corrections, race telemetry, and actual weather cannot alter an earlier snapshot.

Historical validation remains gated on defensible source availability evidence. Software tests can use controlled fixtures; fixture metrics are not real-world model performance. Do not relabel current retrospective downloads as historical as-of data to unblock training.

## Phase 4 result

The feature layer selects the latest known published roster/qualifying and prior-result versions at a UTC cutoff after qualifying and before the race. It excludes current/future event results independently of their timestamps. Optional session, forecast, circuit, pit, reliability, and standings inputs remain missing when unavailable. Current grid values only come from a known roster/grid publication. All numeric features have explicit missingness flags; DNF requires audited Boolean labels rather than guessed status categories. Practice sources remain separate.

Each deterministic Parquet snapshot records a conservative availability bound, content hashes/evidence references, event and prior-race completion times, feature version, form window, and source policy. The `build-snapshot` CLI reads a versioned JSON manifest, verifies declared source kinds and exact local Parquet file hashes, and rejects workspace-escaping paths or mismatched files. The [feature dictionary](FEATURE_DICTIONARY.md) documents definitions, limitations, and the explicit evidence-verification trust boundary. No code automatically promotes unknown retrospective availability into historical publication time.

Verification: 139 offline tests passed, including feature calculations, row-order invariance, required/unknown/late inputs, later corrections, event exclusion, race-session and observed-weather rejection, archived-release evidence, source preference, manifest tampering, and offline CLI-to-Parquet integration. Ruff lint/format and strict mypy passed. Empty qualifying and unknown tyre-compound edge cases found in review are covered by regression tests.

The remaining roadmap now resolves this availability gap through explicit certification, prospective collection, and tiered benchmarks before baseline evaluation. The existing retrospective 2025 downloads alone are not Gold evidence. No fixture-only accuracy claims or arbitrary historical timestamps will be substituted.

## Phase 5 result

Availability evidence now has five explicit classes and binds UTC availability metadata to exact logical table hashes. Gold, Silver, and Development tiers propagate through event and input provenance into each generated feature. Latest-known versions are selected before certification: a newer uncertified required version blocks the snapshot, and a newer uncertified optional version remains missing rather than falling back to an older certified value. Live capture evidence also bounds row capture times. FIA document metadata preserves official identity, hash, publication timezone/time, revision status, and explicit audit references. Weather initialization and availability remain distinct.

Feature snapshots support post-qualifying, provisional-grid, and pre-race windows. Cancellation leaves qualifying classification missing, while pit-lane starts carry no grid ordinal. Historical reliability uses the audited DNF taxonomy and preserves DNS/DSQ as distinct unlabeled states. FastF1/OpenF1 arbitration selects one provider per session, reports disagreement, and quarantines conflicting sessions in certified builds. Practice and tyre fields have conservative aliases that describe observed summaries rather than race pace or degradation.

Prospective capture infrastructure freezes bounded fresh official API payloads into immutable, hash-verified bundles. Its CLI rejects past cutoffs, result/actual-weather/race-session endpoints, malformed requests, and session payloads without validated metadata. Exact retries verify the existing bundle. No scheduler is armed, and no missed historic race is represented as a live capture. Captures still require normalization, event crosswalk, and audit work before feature certification.

Verification: 271 offline tests passed. Ruff lint and format checks passed; strict mypy passed for 33 source files. Leakage review regressions cover required and optional corrections, provider disagreements across evidence tiers, contradictory archive timestamps, capture-bound weather rows, cancellation, and collection-plan authorization. Historical coverage limitations remain: existing retrospective downloads are Development unless exact-version availability can be audited. Phase 6 reports unsupported/empty Gold and Silver tiers honestly.

## Phase 6 result

The benchmark builder reads workspace-contained Parquet files after verifying catalog SHA-256 values. It validates feature schemas, per-row evidence tier and cutoff bounds, exact field completeness, final audited label taxonomy, one winner per race, and label publication after prediction time. It writes deterministic Gold, Silver, and Development Parquet datasets, a machine-readable hash manifest, and coverage for each race/cutoff with evidence quality, missing feature counts, and exclusion reasons. Label columns are prefixed and listed separately from predictive feature columns. Legacy feature schemas remain Development.

Verification: the full suite passed 275 offline tests, including four benchmark builder tests for all tiers, deterministic reruns, hash tampering, incomplete field joins, early labels, and local race discovery. Ruff lint and format checks passed; strict mypy passed for 35 source files. The local scan found 24 normalized 2025 races. There are zero registered feature snapshots and zero final audited outcome files, so all three datasets are valid typed empty Parquet files and all 24 race windows are excluded with explicit reasons. No historical evidence or outcome labels were synthesized.

## Phase 7 implementation

Scikit-learn 1.9.1 documentation was reviewed before integration: `LogisticRegression`, `SimpleImputer`, `Pipeline`, log loss, Brier score, and calibration APIs. The official `TimeSeriesSplit` documentation describes equally spaced observations, which does not fit driver rows grouped into Formula 1 race events. The evaluator instead forms complete race/cutoff cohorts and custom chronological rolling origins.

Every training row must come from an earlier event and have `label_available_at <= test_prediction_timestamp`. All drivers in a race/cutoff remain together. Imputation and scaling fit within each training fold. Heuristics cover winner, podium, DNF, and grid-based finishing order. Logistic baselines cover winner, podium, and observed DNF labels; Ridge regression supplies a finishing-order baseline. Winner probabilities sum to one and podium marginals are projected to exactly `min(3, field_size)` while bounded in [0, 1]. Missing DNF targets remain excluded. Insufficient classes or event counts produce explicit statuses.

Official references: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html), [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [log loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html), [Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html), and [calibration curve](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html).

## Phase 7 result

The `backtest` command evaluates each evidence tier independently. It groups every driver's rows by event, prediction timestamp, and cutoff kind; test events never appear in their training sets. Training rows require final label availability no later than the test prediction time. Numeric imputation and scaling fit only on each chronological training fold. Deterministic seed and estimator settings, benchmark dataset/manifest hashes, fold membership, skipped cohorts, metrics, and out-of-fold prediction Parquet are recorded under ignored model/prediction paths.

Heuristic and logistic baselines produce winner, podium, and DNF probabilities. Winner probabilities are normalized per event; podium marginals are bounded and sum to the available podium places. A grid-based ordering heuristic and Ridge finish-position baseline report MAE. Evaluation includes log loss, Brier score, calibration bins, winner top-1/top-3 accuracy, and finish MAE. Unknown DNS/DSQ DNF labels stay out of DNF fitting and scoring. Empty classes and undersized histories produce `insufficient_data` task or cohort results.

Verification: 280 offline tests passed. Ruff lint and format checks passed; strict mypy passed for 37 source files. Rolling-fold tests verify grouped races and delayed-label gating; deterministic model smoke tests verify probability coherence and Development claim labelling. The real CLI flow built current benchmarks and returned `insufficient_data` for Gold with zero prediction rows and no accuracy metrics. Local coverage still has 24 excluded races and no registered final audited labels or feature snapshots. Fixture results are test evidence only, not real-world accuracy.

## Phase 8 result

The `compare-models` command evaluates bounded histogram boosting, XGBoost,
LightGBM, and CatBoost configurations with the Phase 7 chronological protocol.
An earlier complete event calibrates DNF logits and joint race-order temperature;
fit labels must already be known at that calibration event's cutoff. Imputation
fits only on earlier fit events. A fold edge case was tightened so one unavailable
driver label excludes its entire training race, preserving complete event groups.

One seeded joint simulation produces coherent winner, podium, DNF, and full finish
distributions. Reports include calibration, ranking, position, and probability
metrics on paired baseline/candidate cohorts, with explicit regression lists.
Selection stays separate by cutoff kind and requires 25 paired Gold test races
without baseline regressions under the frozen protocol. Any selection is provisional until future independent
confirmation. Model artifacts retain fit/calibration membership, parameters,
hashes, code and library versions, and actual training devices. Dataset and
coverage hashes are verified before fitting. See [modeling](MODELING.md).

Hardware checks found RTX 3050 Laptop GPU with 4 GiB, driver 616.56, driver CUDA
API 13.4, toolkit 12.9.41, and toolkit runtime API 12.9. Reviewed binary wheels
installed XGBoost 3.4.1, CatBoost 1.2.10, and LightGBM 4.7.0 as optional backends.
XGBoost's CUDA 13.3 build and CatBoost passed actual GPU training probes.
LightGBM's wheel lacks both CUDA and OpenCL GPU support and remains CPU. No
driver/toolkit changes or custom builds were made. On controlled 32,000-row,
32-feature, 80-tree timing workloads, GPU fits were approximately 28% faster for
XGBoost and 46% faster for CatBoost; smaller 8,000-row improvements did not meet
the 15% automatic-use threshold. CPU fallback remains available and normal tests
do not require CUDA or optional boosting libraries.

Verification: 300 offline tests passed. All four installed backends completed CPU
fixture smoke comparisons; histogram boosting prediction reruns and artifact
reloads were deterministic. Ruff lint/format and strict mypy passed for 41 source
files. The real Gold CLI verified its benchmark and wrote `insufficient_data`,
zero predictions, no accuracy metrics, and deferred selection. There are still
24 excluded local races and no registered audited snapshot/outcome joins. No
fixture metrics or synthetic timing workloads are claimed as real-world accuracy.
Phase 8 software is delivered; the selected real-world race model remains a
data-dependent acceptance gate before Phase 9 uses it.

## Dual-track Gold and Phase 8 continuation result

Automatic discovery resolves exact FIA season/event selectors and ranks a bounded
17-race pool from retained current schedules and publication registries without
asserting Gold eligibility. Direct-document auditing then checks a reviewed
nine-race winter shortlist and stops at eight eligible post-qualifying races:
2025 China, Sao Paulo, Las Vegas, Qatar and Abu Dhabi; 2026 Australia, China and
Japan. The cohort contains 166 drivers. Exact qualifying, roster, schedule and
final-target sources have separate hashes, publication bounds, evidence manifests,
availability matrices, inclusion/exclusion reasons and immutable snapshots.
Unverifiable optional grid, weather, practice and historical aggregates remain
missing. Withdrawn preliminary audits remain preserved with explicit reasons.

The Gold baseline backtest evaluated six chronological races with 126 prediction
rows. All four stronger backends completed five identical paired test races,
reserving earlier training and calibration history: 106 drivers per backend,
424 prediction rows. Winner metrics have five race observations, podium metrics
106 driver observations, finishing-position metrics 89 known ordinals, and DNF
metrics zero audited labels. All folds used CPU under the verified device policy.
No champion is selected: eight eligible races permit evaluation, but five paired
tests, missing DNF metrics and recorded baseline regressions fail selection.
The initial historical audit stops here; future independent data must confirm
performance. See [Gold workflow](GOLD_WORKFLOW.md) for hashes and limitations.

Sigmoid, isotonic and identity calibration now reserve configurable earlier event
windows and skip with explicit counts when unsupported. Missing qualifying values
stay null, including an entered driver's all-missing numeric row; Gold requires
exact event, roster and qualifying context rather than invented values.
Model reports preserve the explicit unvalidated DNF prior and strict-JSON backend
parameters. The current 25 paired-race selection floor cannot be reduced or bypassed by
duplicate cutoffs, Silver, Development or fixture metrics.

The limited-user Windows task `f1_ml_predictor_prospective` is installed but
disabled after it caused terminal popups. Manual ticks discover fresh
schedules, freeze post-qualifying versions, normalize only frozen bytes, and later
audit final FIA outcomes separately before benchmark eligibility. A crash releases
the OS advisory lock; captured features cannot be overwritten or backfilled.
The laptop must remain awake and the user logged in. Renew the one-year task in
September 2027. No prospective race snapshot exists before qualifying is observed.

Verification: 495 data/model/collection tests passed, including bounded candidate discovery,
exact historical sources and targets, immutable replay, crash-safe collection,
calibration leakage and Gold selection guards. An additional 35 Phase 9 fixtures
passed during integration. Ruff lint/format and strict mypy passed. Source punctuation,
diff/artifact and commit-message checks are required before the milestone commit.

Remaining Phase 8 acceptance requires at least 15 distinct paired Gold test
races for preliminary comparison and 25 for provisional model selection,
auditable DNF targets and no required baseline regressions, followed by
future independent confirmation. Real Phase 9 WDC/WCC claims remain gated on that
validated race model. Phase 10 can later expose published reports with tier,
sample-count and freshness states; it cannot present these diagnostics as
validated championship forecasts.

## Expanded Gold and rolling evaluation result

The expanded official-document audit includes 25 races and 506 driver-race
observations from 24 unique drivers. Two unresolved 2025 rounds remain excluded.
The immutable Gold benchmark and its source hashes remain available. A separate
`gold-rolling-v1` benchmark derives 3, 5 and 10 race features only from complete
same-season audited prior-round windows whose final labels were available by the
target cutoff. Its feature values, missingness and source-outcome hashes are
frozen separately. Finish means have 299, 219 and 60 populated driver-race rows;
DNF rates have zero, reflecting the source labels rather than an imputed rate.

The baseline evaluates 23 chronological test races. All four stronger backends
evaluate the same 22 paired test races with 1,784 total prediction rows. The
preliminary comparison gate is open, while the primary accuracy claim gate and
model selection remain closed. Each backend has documented baseline regressions;
paired race-level uncertainty is reported against both baselines. CPU was used
for the real cohort under the measured-benefit hardware policy. See
[Gold workflow](GOLD_WORKFLOW.md) for commands and source limitations.

The retained FIA finals provide 492 unknown, eight DNS and six DSQ statuses.
None support a binary DNF label under the existing taxonomy. See the
[DNF evidence audit](DNF_AUDIT_STATUS.md). The collector now records missed
races discovered after an outage and bounds retry backoff. A windowless Python
task runner writes local JSONL logs; the existing interactive task stays disabled.
The replacement task has been previewed with a startup trigger and wake setting.
Windows credential entry and a scheduled network tick remain to activate it.

The next audit admitted 2026 Azerbaijan, Hungary, Belgium and Great Britain
under the unchanged Gold policy, reaching 29 races, 594 driver-race observations
and 24 unique drivers. Italy and the Netherlands remain excluded pending later
document review. A separate binary DNF outcome version has 547 known labels:
67 retired and 480 finished; DNS, DSQ and ambiguous cases remain unknown. The
versioned rolling comparison evaluates 26 identical paired test races across
the same four backends, including 494 known DNF test labels. The original
25-race result above is retained as a historical result.

Race-level paired diagnostics, two controlled feature ablations and calibration
bins identify podium tail failures, sparse early fitting windows and extensive
optional-feature missingness. They do not change the frozen selection protocol.
Every backend still has a recorded baseline regression, so `no_selection`
remains the explicit decision. See the [current Gold regression audit](BASELINE_REGRESSION_AUDIT.md)
and [binary DNF audit](DNF_AUDIT_STATUS.md). Continue future Gold collection,
podium tail and calibration investigation, and credential-backed unattended
collection verification. Do not issue validated WDC or WCC forecasts.

## Phase 9 simulation infrastructure result

The offline championship simulator consumes complete joint sampled orders from
the shared race sampler. It resamples whole orders, preserving within-race driver
dependencies. Event-specific constructor assignments preserve transfers without
reallocating old team points. Scoring is explicit by season, race/sprint and
distance case; 2019-2024 fastest-lap bonuses require explicit eligibility, and
2025 onward rejects that bonus. Points classification is supplied separately
because a sampled DNF does not establish FIA eligibility or a retirement ordinal.

Fixed seeds, canonical event ordering, exact rational points, unresolved countback
tie mass, JSON serialization and source/input hashes make engineering runs
reproducible. Default outputs are `engineering_only`. A validated scenario requires
explicit provenance for a validated Gold model from at least eight independent races,
compatible model IDs, pre-cutoff evidence, earlier validation rounds and explicit
classification samples. This checks a provenance contract; it does not establish
championship calibration. Remaining events are independent and season-wide shared
effects are omitted. See [simulation assumptions](SIMULATION.md).

Verification: 39 deterministic simulation tests and the complete 534-test offline
suite passed. Ruff lint and format checks and strict mypy passed for 54 source
files; diff, generated-artifact and forbidden-punctuation reviews passed before
commit. Fixtures are engineering checks, with no real WDC/WCC probability claim.
Validated models, championship calibration, explanations and dashboard integration
remain on the roadmap rather than being marked complete by fixture tests.

## Orchestration

The main agent owns architecture, phase integration, ML choices, final review, and git operations. Bounded independent work can be delegated for current API documentation, isolated source clients after contracts exist, test fixtures, and leakage review. Agents should receive only the relevant interface and should avoid overlapping edits. Mechanical checks remain deterministic scripts.

## Current decisions and open risks

- Phase 1 uses standard-library contracts; Phase 2 adds HTTPX and PyArrow. PyArrow does not ship typing markers, so its imports have a scoped mypy override and table shapes are covered by schema tests. A lockfile can be added once the runtime dependency set stabilizes.
- Store source publication time where available. For older records without it, strict historical backtests must use conservative availability rules or exclude that field.
- Race calendar changes, sprint format changes, points rules, penalties, and constructor/driver transfers require season-aware handling in later phases.
- Forecast archives, FastF1 coverage, and FIA document access may limit historical backtest depth. Report coverage explicitly rather than silently imputing future data.
