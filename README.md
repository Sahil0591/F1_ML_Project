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

## Graphify

Graphify is configured through a git post-commit hook. After each commit, the hook updates the local knowledge graph outputs in `graphify-out/`.
