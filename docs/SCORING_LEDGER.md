# Audited championship scoring ledger

The scoring importer accepts awarded points only. Race finish positions and present-day standings are never converted into historical points. The committed FIA audit in `data/audit/scoring_rules.json` and `data/audit/event_points_evidence.json` uses `scoring-evidence-v1`; `docs/SCORING_EVIDENCE_AUDIT.md` explains its sources, publication clocks, disputed values, and amendments. The native importer verifies retained FIA document and regulation bytes, the collection index hash, driver awards, and constructor matrix reconciliation. It reconstructs a complete event version at each FIA points publication clock. A revision only affects cutoffs after its publication. Pending appeals remain missing.

## Handoff format

The compact handoff and synthetic fixture format below uses `schema_version: 1`. The committed native FIA format uses `schema_version: "scoring-evidence-v1"` and is parsed separately. In the compact format, unknown or disputed values remain `null` with `revision_status: "unknown"` or `"disputed"`; they cannot populate Gold point features. Every compact object has exact keys. Extra or missing keys fail import.

`scoring_rules.json` is `{"schema_version":1,"rules":[...]}`. Each rule has:

| Field | Meaning |
| --- | --- |
| `season`, `first_round`, `last_round` | Inclusive season and round interval. Intervals may not overlap. Use a new rule for a midseason change. |
| `race_points`, `sprint_points` | Ordered points by classified position. Empty sprint list means no sprint awards. |
| `reduced_race_points` | List of ordered schedules for shortened races. Empty list means none. |
| `fastest_lap_points`, `fastest_lap_eligible_through` | Bonus value and eligibility limit, or null limit. The audit must verify actual eligibility. |
| `constructor_scoring` | `sum_awarded_entries` or `not_contested`. The ledger records awards to constructors explicitly, including exclusions and penalties; seasons without a constructor championship stay missing. |
| `source_evidence`, `evidence_hash`, `revision_status` | Source references, canonical object SHA-256, and `audited`, `revised`, `unknown`, or `disputed`. |

`event_points_evidence.json` is `{"schema_version":1,"events":[...]}`. Each event version has `season`, canonical `event_id`, `completed_at`, `effective_at`, `race_schedule`, `complete`, `revision_status`, `expected_driver_ids`, `entries`, and `evidence_hash`. `race_schedule` is `standard` or `reduced:<zero-based index>` into the season rule's reduced schedules. An award is checked against that event's selected schedule. A revision is a complete replacement event version, not a delta. `expected_driver_ids` lists the audited starters or classified entrants and must match the driver and constructor entries when `complete` is true. An event with any unknown awarded amount remains missing in historical snapshots.

Each entry has `season`, `event_id`, `driver_id`, `constructor_id`, `race_points`, `sprint_points`, `bonus_points`, `adjustment_points`, `total_points`, `effective_at`, `source_evidence`, `evidence_hash`, and `revision_status`. The four components must sum to the total when known. Use a normal driver and constructor entry for points awarded to both. A constructor-only adjustment has `driver_id: null` and zero race, sprint, and bonus components. A driver-only adjustment has `constructor_id: null`. A DSQ or revised classification is represented by a new full event version effective when the revision became known. No current classification can overwrite an earlier cutoff.

`source_evidence` is a nonempty list of `{"reference":"...","sha256":"64 lowercase hex digits"}`. References should identify the exact official document or retained audit artifact. `evidence_hash` is SHA-256 of the JSON object without its `evidence_hash` key, serialized with sorted keys, compact separators, UTF-8, and no NaN. The importer verifies object hashes and records the raw audit file hashes. Source document hashes are preserved for the separate source audit. Publication timestamps use ISO 8601 UTC. Point amounts use quarter-point precision. Evidence references and content hashes are retained in scoring provenance.

## Build and evaluation

Run `python -m f1_ml_predictor build-gold-scoring --benchmark-dir <immutable historical enrichment directory>` with the committed FIA audit files and the retained local FIA source cache described in [the evidence audit](SCORING_EVIDENCE_AUDIT.md). The raw cache is ignored by Git, so another checkout must restore the exact hashed source bytes before import. The command validates the schemas, retained source hashes, and source benchmark hashes. It writes `data/benchmarks/gold_championship_scoring_v3/<source manifest hash>/<ledger hash>/` with a version 4 manifest, copied secondary tiers, a new Gold Parquet file, season coverage, missing reasons, and per-row scoring provenance. The source manifest and prior model runs remain unchanged. Rebuilding the same inputs must produce the same files or fail on an immutable collision.

Only complete earlier rounds with an audited rule and event version effective before the prediction cutoff contribute to a snapshot. Missing earlier rounds leave all point features missing. Driver totals follow the driver across constructor transfers. Constructor totals use the constructor credited in each event entry. Rolling features need all 3, 5, or 10 earlier rounds. Tied championship positions remain missing when countback evidence conflicts or needs further FIA resolution; points and leader gaps remain available. Four 2022 United States race placings disagree between the FIA points matrix and audited classification, so their countback contribution is withheld for affected ties.

The baseline and joint model runners accept version 4 benchmarks and record `scoring_ledger_sha256` in run metadata. The scoring benchmark is evaluated with the frozen chronological Gold protocol. Driver, constructor, and combined point ablations use the same folds and race-paired uncertainty. Preserve `no_selection` unless the existing selection gates are met. These standings features alone do not validate WDC or WCC forecasts.

## Constructor reliability

The existing historical enrichment computes `constructor_dnf_rate` only from prior hash-verified Gold outcomes and explicit known binary DNF labels. It records an observation count and leaves the rate missing when none are known. That path is separate from championship scoring. The current Gold binary DNF coverage remains sparse, so this feature should be evaluated separately rather than used as point evidence.
