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

The offline tests use mocks. The project does not produce predictions yet;
historical publication times remain unknown and must be resolved before strict
point-in-time feature use.

## Graphify

Graphify is configured through a git post-commit hook. After each commit, the hook updates the local knowledge graph outputs in `graphify-out/`.
