# Data Trust and As-Of Certification

Certification describes evidence, not model accuracy. A current API response does
not prove that its values existed at an earlier prediction cutoff. Final audited
outcomes can be labels after a race; predictive inputs must already be available.

## Evidence and Tiers

`AvailabilityEvidence` binds an exact logical table hash to a reference, UTC
availability bound, and evidence class. File manifests additionally bind exact
Parquet bytes. These checks detect mismatches, not fraudulent curator attestations.
Independent review of supplied archive/publication evidence remains required.

| Class | Required evidence | Tier |
| --- | --- | --- |
| `captured_live` | Exact capture time equals availability; exact artifact hash | Gold |
| `source_published_timestamp` | Exact value version and publication timestamp; explicit audit | Gold when audited, otherwise Development |
| `versioned_archive` | Immutable version identity, availability bound and explicit audit | Gold when audited, otherwise Development |
| `conservative_reconstruction` | Documented reconstruction method, conservative upper availability bound and explicit audit | Silver when audited, otherwise Development |
| `current_state_only` | Current or unknown historical state | Development |

The weakest used input determines a snapshot's tier. A legacy free-form reference
never upgrades itself. Missing optional inputs do not lower a tier, but missingness
and exclusion reasons remain recorded. Gold accuracy claims require Gold benchmark
data. Silver and Development support labelled engineering and exploration only.

Latest-known selection precedes certification. A newer uncertified correction to a
required input blocks a certified snapshot. A newer uncertified optional input is
omitted without reverting to an older certified value. Known provider disagreement
is retained even if the alternative provider has weaker evidence.

## Prediction Windows

`post_qualifying` requires qualifying completion or a published cancellation decision.
`provisional_grid` additionally requires explicit provisional/final grid status.
`pre_race` defaults to the final 60 minutes before the known start. Every cutoff
must precede the race. Separate windows are separate evaluation cohorts.

Cancelled qualifying leaves classification features missing. An explicit pit-lane
start has no numeric grid ordinal. Grid zero does not imply pit lane. No grid is
invented from qualifying or eventual race results.

## FIA, Weather and Session Audits

FIA evidence retains official URL, document identity/hash, printed time and timezone,
provisional/final/revised/recalled status, and normalized table hash. The curator
must inspect the exact document and explicitly attest its contents. Recalled
documents cannot certify features. CET and CEST are explicit, not guessed from a
date. There is no invented FIA JSON API or automatic PDF extraction.

Weather retains forecast initialization, actual availability, capture time and
target time separately. Initialization or archive presence alone cannot certify
release. Open-Meteo's model-update metadata distinguishes initialization from
availability; retrospective release evidence must be independently supported.
Observed race weather is not a pre-race forecast.

For historical Gold enrichment, FIA practice and grid values require a matching
retained document-registry row, its publication time, exact PDF hash and cover,
an unambiguous roster crosswalk, and a bound no later than the target cutoff.
These inputs remain missing if the latest eligible version is recalled or cannot
be parsed conservatively. FIA race result ordinals do not encode awarded points:
season-specific sprint rules, fastest-lap awards, reduced points, and revisions
require a separately audited scoring ledger. Historical Open-Meteo run archives
do not establish per-run release times; retrospective FastF1 and OpenF1 practice
fetches do not establish exact historical availability on their own.

FastF1 and OpenF1 are arbitrated per session, by evidence and stated preference.
Their incompatible filters are not averaged. Best-lap disagreement above two
percent is flagged; certified builds quarantine those session features. FastF1
generated/deleted/inaccurate/pit laps are excluded. Missing OpenF1 quality flags
are not proof of clean laps. Practice summaries are observations, not fuel-adjusted
race pace, stint predictions, or tyre degradation estimates.

## Audited Outcomes

`audited-dnf-v1` separates finished, mechanical retirement, incident retirement,
other retirement, DNS, disqualification, and unknown. Retirement maps to DNF;
finished maps to non-DNF; DNS/DSQ/unknown retain null DNF labels. Classification is
independent of retirement. Raw status text is retained but is not a classifier.
Final label publication time and audit reference are mandatory. Training later
must respect label availability, even when retrospective evaluation uses labels.

## Prospective Collection

Run `capture-weekend <plan.json> --root <workspace>` during a supported window.
It freezes fresh bounded official JSON responses under `data/raw/prospective`.
Supported endpoints are schedule/qualifying, completed practice/qualifying session
metadata/laps/stints/pits/drivers, and forecast weather. Race results, race sessions,
actual weather, and telemetry are excluded. One process shares each API budget.

Example plan structure (replace event times and request selection before running):

```json
{
  "version": 1,
  "season": 2026,
  "round": 1,
  "race_start": "REPLACE_WITH_KNOWN_UTC_RACE_START",
  "qualifying_decision_at": "REPLACE_WITH_KNOWN_UTC_DECISION_TIME",
  "cutoff_kind": "post_qualifying",
  "requests": [
    {
      "name": "qualifying_payload",
      "role": "qualifying",
      "url": "https://api.jolpi.ca/ergast/f1/2026/1/qualifying"
    }
  ]
}
```

The collector records the actual end-of-capture cutoff, never a requested old
timestamp. A bundle slot cannot change; exact retries are verified and reused.
`verify-capture <bundle>` checks manifest, paths, table hashes and file hashes.
Raw collection does not automatically establish a complete roster/grid certificate.
Normalization, crosswalk and publication audits are still necessary before features.
No scheduler is armed yet. Infrastructure is ready for future weekends, but missed
historical windows cannot be recovered as captured-live evidence.

## Verified Primary Documentation

- [FIA document index](https://www.fia.com/documents): document versions and publication evidence.
- [FIA 2026 sporting regulations](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_06_-_2026-04-28.pdf): classification and sporting rules.
- [Open-Meteo model updates](https://open-meteo.com/en/docs/model-updates) and [single runs](https://open-meteo.com/en/docs/single-runs-api): initialization and availability are different times.
- [FastF1 Laps](https://docs.fastf1.dev/core.html#fastf1.core.Laps) and [OpenF1 laps](https://openf1.org/docs/#laps): provider-specific quality fields and limitations.
