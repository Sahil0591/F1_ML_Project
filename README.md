# F1 ML Project

This repository is building point-in-time Formula 1 race and championship predictions.
The first prediction snapshot is after qualifying and before the race. See the
[project plan](docs/PROJECT_PLAN.md) for data sources, architecture, phases, and
acceptance criteria.

## Local setup

Use Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
```

## Historical ingestion

```powershell
.\.venv\Scripts\python.exe -m f1_ml_predictor ingest-season 2025
```

Historical data is cached under `data/raw/jolpica`; typed Parquet tables go to
`data/normalized`. Reruns skip unchanged historical data. Use `--refresh` to
recheck an older season; current and future seasons refresh automatically.
Run one ingestion process at a time to share the API request budget.

## Session and forecast enrichment

```powershell
.\.venv\Scripts\python.exe -m f1_ml_predictor list-openf1-sessions 2025
.\.venv\Scripts\python.exe -m f1_ml_predictor ingest-openf1-session 2025 1 9689
.\.venv\Scripts\python.exe -m pip install -e ".[dev,fastf1]"
.\.venv\Scripts\python.exe -m f1_ml_predictor ingest-fastf1-session 2025 1 Q
```

Ingest the Jolpica season first to establish event and driver-number crosswalks.
OpenF1 session keys come from API discovery. Only completed practice and qualifying
sessions are accepted. FastF1 loads lap timing and race-control metadata, not car
telemetry. Cached responses and content-addressed Parquet summaries are reused.

`capture-forecast <season> <round>` captures Open-Meteo weather for an event within
the next seven days. Archived model runs require explicit release-time evidence
through the Python client; model initialization alone is not publication proof.
FIA document URLs and publication times can be recorded with `record_fia_evidence`.

Normal tests use mocks and do not require external APIs. The project does not
produce predictions yet. Retrospectively fetched sessions keep their capture-time
availability; historical Jolpica publication times remain unknown. Strict backtests
must exclude unsupported historical inputs rather than backdate them.

## Pre-race features

`build-snapshot <manifest.json>` creates deterministic Parquet features from verified
as-of input versions. It rejects unknown/late required publications, excludes current
and future race results, and leaves unavailable optional features missing. See the
[feature dictionary](docs/FEATURE_DICTIONARY.md) for input contracts and manifest fields.

An evidence manifest binds publication times to exact file hashes. It is not a way to
backdate present-day downloads. Real training/backtests await verified historical
snapshots or a prospective archive; offline fixture tests are not model-performance
evidence.

## Data Trust and Prospective Collection

Evidence classes and Gold/Silver/Development certification now propagate into
feature snapshots. Audited DNF categories, provider arbitration, explicit cutoffs,
pit-lane starts and cancelled qualifying preserve provenance and missingness.
See [data trust](docs/DATA_TRUST.md) for the audit contract and capture-plan format.

`capture-weekend <plan.json>` freezes fresh pre-race API payloads;
`verify-capture <bundle>` checks their integrity. No scheduler is armed and no past
race is relabelled as a live capture. Benchmarks and baseline evaluation follow;
historical evidence coverage remains the gate for primary accuracy claims.

## Benchmark Datasets

Run `build-benchmarks` to scan locally ingested race partitions, or pass
`--catalog <catalog.json>` with feature/outcome Parquet paths, exact SHA-256 values,
prediction timestamps, and cutoff kinds. The command writes Gold, Silver, and
Development datasets plus `manifest.json` and `coverage.json` under
`data/benchmarks`. Gold is the primary accuracy tier. Current local coverage is
zero included and 24 excluded races: no feature snapshots or final audited outcome
files are registered. The builder does not upgrade retrospective rows to fill them.

## Baseline Backtests

Run `backtest --tier Gold` after building benchmarks. The evaluator creates rolling
race/cutoff cohorts, fits imputation and logistic/regression pipelines within each
training fold, and gates training rows on audited label publication time. It writes
metrics under `models/backtests` and out-of-fold predictions under
`data/predictions/backtests`. Empty or undersized tiers return `insufficient_data`
without accuracy metrics. Silver and Development results remain exploratory;
fixture checks are not performance claims.

## Graphify

Graphify is configured through a git post-commit hook. After each commit, the hook updates the local knowledge graph outputs in `graphify-out/`.
