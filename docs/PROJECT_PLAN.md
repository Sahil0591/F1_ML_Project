# F1 ML Predictor Project Plan

Status: Phases 1 through 5 complete; Phase 6 evidence-tiered benchmark datasets are next. Updated 2026-09-28.

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
| 8. Stronger probabilistic race models | Depends on 7. Compare gradient boosting, XGBoost, LightGBM, and CatBoost where justified, then calibrated coherent winner/podium/DNF/finish distributions. | Same chronological protocol; calibration, ranking, position and DNF tests; no undocumented baseline regression. | Risk: overfit and incoherent probabilities. Output: selected probabilistic race model. |
| 9. Championship simulation and explanations | Depends on 8. Monte Carlo WDC/WCC with current known points, season-aware rules, local/global explanations and SHAP where compatible. | Fixed seeds, points fixtures, probability sums, sensitivity, explanation provenance. | Risk: changing rules and compounded uncertainty. Output: championship probabilities and explanations. |
| 10. Dashboard and operations | Depends on 7 for initial interface and 9 for full scope. Separate read-only API/UI for published predictions, evidence tier, uncertainty and freshness; scheduled bounded refresh. | API/UI integration, responsive workflows, stale/missing states, deployment/runbook, offline suite. | Risk: presenting exploratory or stale outputs as reliable. Output: dashboard and operational collection workflow. |

## Remaining roadmap policy

Phase 5 evidence classes are `captured_live`, `source_published_timestamp`, `versioned_archive`, `conservative_reconstruction`, and `current_state_only`. Evidence must bind to the exact value version. Captured-live availability equals capture time; published/archive evidence requires independently audited timestamp/version metadata; conservative reconstruction requires an explicit method and upper availability bound. Existing free-form references default to current-state/Development, never upgraded implicitly.

Gold requires verified direct pre-cutoff evidence for every used sensitive input. Silver permits audited conservative reconstruction with documented assumptions. Development contains legacy/current-state/unaudited exploratory inputs and cannot support primary accuracy claims. Missing optional features remain missing and visible in coverage. Final audited outcomes are labels, not predictive inputs; their later publication is allowed for evaluation but training folds must respect label availability.

Prospective collection freezes exact fresh API payloads and derived input versions with UTC capture times and hashes. Named post-qualifying, provisional-grid, and pre-race windows are separate prediction cohorts. A frozen cutoff cannot be overwritten or retrospectively populated. FIA audit metadata retains exact document identity, printed publication time/timezone, revision/recall status, and hash. Weather initialization is separate from actual availability. FastF1/OpenF1 arbitration selects a documented provider per session and flags disagreement, never averages incompatible filters.

Phase 6 will run against available local data without manufacturing Gold/Silver races. It must emit valid empty tiers and explicit exclusion reasons when evidence or audited labels are absent. Baseline software can be developed on deterministic fixtures and clearly labelled Development data while certified coverage accumulates; fixture metrics are not real-world accuracy.

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

Verification: 271 offline tests passed. Ruff lint and format checks passed; strict mypy passed for 33 source files. Leakage review regressions cover required and optional corrections, provider disagreements across evidence tiers, contradictory archive timestamps, capture-bound weather rows, cancellation, and collection-plan authorization. Historical coverage limitations remain: existing retrospective downloads are Development unless exact-version availability can be audited. Phase 6 must report unsupported/empty Gold and Silver tiers honestly.

## Orchestration

The main agent owns architecture, phase integration, ML choices, final review, and git operations. Bounded independent work can be delegated for current API documentation, isolated source clients after contracts exist, test fixtures, and leakage review. Agents should receive only the relevant interface and should avoid overlapping edits. Mechanical checks remain deterministic scripts.

## Current decisions and open risks

- Phase 1 uses standard-library contracts; Phase 2 adds HTTPX and PyArrow. PyArrow does not ship typing markers, so its imports have a scoped mypy override and table shapes are covered by schema tests. A lockfile can be added once the runtime dependency set stabilizes.
- Store source publication time where available. For older records without it, strict historical backtests must use conservative availability rules or exclude that field.
- Race calendar changes, sprint format changes, points rules, penalties, and constructor/driver transfers require season-aware handling in later phases.
- Forecast archives, FastF1 coverage, and FIA document access may limit historical backtest depth. Report coverage explicitly rather than silently imputing future data.
