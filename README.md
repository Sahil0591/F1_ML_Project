# F1 ML Project

This repository is building point-in-time Formula 1 race and championship predictions.
The first prediction snapshot is after qualifying and before the race. See the
[project plan](docs/PROJECT_PLAN.md) for data sources, architecture, phases, and
acceptance criteria.

## Local setup

Use Python 3.11 or newer. From the repository root:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

Phase 1 provides package and point-in-time contracts. It does not fetch live
data or produce predictions yet.

## Graphify

Graphify is configured through a git post-commit hook. After each commit, the hook updates the local knowledge graph outputs in `graphify-out/`.
