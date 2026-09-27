# F1 ML Predictor Project Plan

Status: Phases 1 through 3 complete; Phase 4 next. Updated 2026-09-28.

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
| 5. Baselines and backtests | Depends on 4. Establish heuristic and logistic winner/podium/DNF baselines, ranking or position baseline, rolling evaluation, calibration reports. | Model smoke and rolling-fold tests; event grouping; probability normalization; saved metrics and reproducible seeds. | Risk: small sample and class imbalance. Output: defensible benchmark and evaluation report. |
| 6. Probabilistic race models | Depends on 5. Compare gradient boosting, XGBoost, LightGBM, and CatBoost only where justified. Calibrate winner, podium, DNF, and finishing distributions with coherent event-level outputs. | Same backtest protocol; calibration, ranking, position and DNF checks; model manifests and no regression against baselines without a documented tradeoff. | Risk: overfit and incoherent probabilities. Output: selected race model. |
| 7. Championship simulation and explanations | Depends on 6. Simulate remaining events for WDC/WCC using current published points and deterministic Monte Carlo; add global and local explanation summaries, then SHAP if compatible. | Fixed-seed simulation tests, points-rule fixtures, probability sums, sensitivity checks, explanation provenance. | Risk: schedule/rules changes and compound uncertainty. Output: WDC/WCC probabilities and explanations. |
| 8. Dashboard and operations | Depends on 5 for an initial useful interface, 7 for full scope. Expose published predictions and provenance through a read-only service and interactive dashboard; schedule bounded refresh jobs. | API/UI integration tests, responsive flow checks, stale-data and missing-data states, offline test suite, deployment/runbook checks. | Risk: showing stale predictions as current. Output: usable dashboard and refresh workflow. |

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

## Orchestration

The main agent owns architecture, phase integration, ML choices, final review, and git operations. Bounded independent work can be delegated for current API documentation, isolated source clients after contracts exist, test fixtures, and leakage review. Agents should receive only the relevant interface and should avoid overlapping edits. Mechanical checks remain deterministic scripts.

## Current decisions and open risks

- Phase 1 uses standard-library contracts; Phase 2 adds HTTPX and PyArrow. PyArrow does not ship typing markers, so its imports have a scoped mypy override and table shapes are covered by schema tests. A lockfile can be added once the runtime dependency set stabilizes.
- Store source publication time where available. For older records without it, strict historical backtests must use conservative availability rules or exclude that field.
- Race calendar changes, sprint format changes, points rules, penalties, and constructor/driver transfers require season-aware handling in later phases.
- Forecast archives, FastF1 coverage, and FIA document access may limit historical backtest depth. Report coverage explicitly rather than silently imputing future data.
