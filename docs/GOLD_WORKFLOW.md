# Dual-track Gold collection

Historical Core and prospective collection use the existing evidence classes,
feature schema, audited DNF taxonomy, grouped chronological folds and benchmark
builder. A smaller feature set can be Gold. Missing practice, tyres, telemetry,
forecasts and historical aggregates do not disqualify an otherwise audited Core.
Every used input must still pass the unchanged direct-evidence Gold policy.

## Historical candidates

### Five-season exhaustive expansion

`scripts/expand-historical-gold.py` discovers every completed race in the 2022–2026
season window as of the run clock. It does not use a minimum-race stop. The
discovery catalog retains current Jolpica schedule responses and exact FIA event
registry observations; those are leads, not historical feature evidence.

The script replays reviewed candidates and constructs additional 2022–2026
candidates from exact registry publication rows. It resolves both legacy and
modern FIA registry layouts and verifies legacy PDF cover identity before
attempting every race through the unchanged direct-evidence Gold builder.
Official Formula 1 schedule articles and event timetable content bind the race
clock by the selected cutoff. It rebuilds the tiered benchmark and checks
same-season prior-race rolling availability. Unreadable tables, unreviewed later
rulings, conflicting schedule clocks and other failures remain per-race exclusions. Grid,
penalty and weather values are used only with cutoff-valid direct evidence;
optional features stay null otherwise. Metadata presence alone never creates
Gold, Silver or Development rows.

```powershell
.\.venv\Scripts\python.exe scripts/expand-historical-gold.py
```

For an exact replay of an already retained discovery catalog, pass its path with
`--discovery`. The immutable `five-year-coverage-*.json` output under
`data/benchmarks/historical_audit` contains all race decisions, annual feature
coverage, source coverage and exclusion reason counts. The command's summary
prints the output path and tier counts. The 2026 season is partial until later
races finish; the window is five season identities, not five complete calendars.
The [initial expansion audit](HISTORICAL_GOLD_EXPANSION_2026-09-30.md) records
the 36-race baseline. The [legacy FIA recovery audit](LEGACY_FIA_GOLD_RECOVERY_2026-09-30.md)
records 107 candidates, 84 Gold races, and 23 exact exclusions. All three
2026 later-document no-change reviews are bound to retained source hashes.

`discover-gold-candidates` automatically discovers a bounded pool from current
Jolpica schedule identities and exact FIA season/event selectors. It retains raw
schedule and registry responses, ranks publication/version metadata, and writes
immutable catalog and auditability reports. It downloads no PDFs and certifies
zero Gold races. Current schedule identities require an official schedule audit.
Failed selector URLs are requested once per run; uncertainty has explicit reasons.

`HISTORICAL_CANDIDATES.json` records a bounded researched 17-race pool with official
links, raw clocks, document identities, schedule leads and revision caveats.
`audit-candidates` retains registry HTML and qualifying PDFs, SHA256 hashes,
request parameters, retrieval times and provider headers under `data/raw/fia_audit`.
It ranks expected qualifying coverage before reconstruction and writes an
immutable auditability report under `data/benchmarks/historical_audit`.

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[audit]'
.\.venv\Scripts\python.exe -m f1_ml_predictor discover-gold-candidates --seasons 2026 2025 --limit 17
.\.venv\Scripts\python.exe -m f1_ml_predictor audit-candidates docs/HISTORICAL_CANDIDATES.json
```

Discovery never certifies the download as historically available. The FIA
registry's summer CET label and PDF event-local issue clocks differ. PDF issue
time is not release time. A recalled document's generic URL may now serve its
replacement. Verify exact document identity, registry publication and every
potentially earlier revision. Missing recalled bytes can require exclusion.
Winter dates with unambiguous European standard time are a useful second pool.

The official F1 [2025 start-time announcement](https://www.formula1.com/en/latest/article/f1-announces-race-start-times-for-2025-season.490KLLD7T1AAM7wQl28tn6)
includes UTC race times and a printed publication timestamp. A retained exact
version and completed audit must bind event context before it becomes evidence.
Administrative canonical ID crosswalks are explicit; entrant/team values come
from the contemporary classification or entry list, never later team membership.
The reviewed 2026 announcement in the shortlist is checked independently against
its retained publication metadata and schedule table. Neither a claimed clock
nor a schedule URL alone is accepted.

`audit-winter` consumes the reviewed winter shortlist or the expanded 27-race
catalog in `HISTORICAL_EXPANSION_CANDIDATES.json`. It verifies the contemporary roster,
exact qualifying cover identity and values, latest required versions at cutoff,
schedule publication/table, and latest final classification with no unreviewed
later rulings. Minute-resolution publication clocks use the end of the minute
as their conservative availability bound. The printed CET clock on summer dates
uses UTC+1 as the later possible UTC bound if CET or CEST was intended.
Raw target clocks retain their ambiguity and a conservative upper bound.

```powershell
.\.venv\Scripts\python.exe -m f1_ml_predictor audit-winter docs/HISTORICAL_WINTER_CANDIDATES.json
.\.venv\Scripts\python.exe -m f1_ml_predictor audit-winter docs/HISTORICAL_EXPANSION_CANDIDATES.json --minimum-races 25 --reuse-retained
```

`--reuse-retained` reuses hash-verified registry observations less than 24 hours
old with their original retrieval metadata. Registered frozen audits replay their
original evidence, rather than replacing it with a present-day download. The
initial audit stops at eight eligible races. A damaged 2025 Australian PDF has
four visually reviewed, exact-SHA-bound Q1 transcriptions in the shortlist;
they cannot apply to another PDF or overwrite a parsed value. That event was
not needed after the initial coverage target was reached.

The expanded audit first reached 25 eligible races on 2026-09-29: 506 driver-race
observations from 24 unique drivers, with zero audited DNF labels in that
original outcome version. The Dutch
Grand Prix has a later right-of-review decision whose retained exact document
says the served penalty cannot be remedied by amending the final classification.
Its identity and conclusion are checked again during replay. The 2025 Spanish
final classification could not be parsed from its retained PDF, and the Canadian
evidence had an immutable artifact conflict, so neither entered this cohort.

## Reconstructing Core

`build-gold-core` consumes the existing version-2 feature request manifest.
Additional `document_bindings` retain each PDF and its registry path/hash,
official `document_url`, `document_id`, `event_id`, status, `version_audited`,
`latest_at_cutoff_audited` and `audit_reference`. Each `table_bindings` item binds
an evidence `reference`, normalized `table_sha256` and `available_at_utc`.
These are assertions from an actual document/value audit, not flags to enable
without checking the retained sources. Present-day API values cannot substitute
for the document's values. Current-state and conservative reconstruction evidence
remain Development and Silver respectively and are rejected by the Core builder.

```powershell
.\.venv\Scripts\python.exe -m f1_ml_predictor build-gold-core data/raw/fia_audit/request.json
```

Required roster, qualifying and event inputs need direct audited publication or
archive evidence. Richer session inputs are rejected by Core. Optional values
stay null with missingness indicators. The builder verifies persisted Parquet
contents, freezes an evidence manifest and feature availability matrix, and
updates `data/benchmarks/gold_core_registry.json`. Frozen feature requests exclude
targets. A later final outcome attachment requires separate exact final-document
bindings and the complete roster; it cannot replace an existing attachment.
Target bindings retain registry path/hash, `latest_final_audited`, the exact
outcome SHA256, row audit reference and label availability. A failed later audit
can withdraw an index entry with an explicit reason while preserving all frozen
files and the withdrawal record.
The benchmark report records each inclusion or exclusion.

Final labels may be published after the predictive cutoff. Their own availability
still controls when subsequent training folds can use them. Unknown retirement
causes remain unknown under `audited-dnf-v1`.
The original 25-race cause-oriented taxonomy contained 492 unknown, eight DNS
and six DSQ statuses across 506 observations. A later separate
[binary DNF audit](DNF_AUDIT_STATUS.md) cross-checks final outcomes without
changing the frozen source attachments.

## Prospective collection

`collect-next-race` runs one bounded tick, discovers the next event, polls after
scheduled qualifying begins, and waits for a validated nonempty qualifying
response. Scheduled end time alone never proves completion. Captures use the
observation-time decision and actual collector clock. A missed race is not replayed
as a live capture. Default capture is post-qualifying; pre-race is opt-in.

```powershell
.\.venv\Scripts\python.exe -m f1_ml_predictor collect-next-race
.\.venv\Scripts\python.exe -m f1_ml_predictor collection-status
```

The scheduler retains raw discovery JSON, plans, payload hashes and explicit
versions. Captured inputs are immutable bundles. Repeated ticks are idempotent;
`--new-capture` requests a new version. Derived snapshots use only frozen bytes.
Label attachments and registry updates are separate from predictive bundles.
Full eligibility requires exact final audited outcomes and a complete roster.

`scripts/collect-next-race.ps1` writes daily logs under the ignored raw directory.
`scripts/install-collector-task.ps1` prepares one limited-user Windows task named
`f1_ml_predictor_prospective`, every five minutes for one year plus a startup
trigger. It uses `pythonw.exe` without a console, starts when available, and
requests wake to run. Windows needs a password-backed task logon for network
access while signed out ([S4U has no network access](https://learn.microsoft.com/en-us/windows/win32/api/taskschd/nf-taskschd-itaskfolder-registertask));
the installer obtains the credential through the local
Windows prompt and does not print it. Preview the action before installation.
Renew the periodic trigger after one year. Overlapping runs are suppressed.
The scheduler also uses an OS-released advisory lock. A killed process cannot
leave a permanent sentinel lock. Provider errors stay visible in status and
logs, and error ticks return a failing exit code.

After the next-race capture check, each tick can audit one completed frozen event
against the exact FIA final classification. It rechecks the registry after PDF
retrieval and refuses later rulings, recalled/replaced documents, incomplete
rosters or unsupported aliases. Raw final sources and audit manifests are retained
separately, then attached with exact outcome hashes. Unknown DNF causes remain
unlabelled. A capture alone does not establish evaluation eligibility.

Audited outcomes can later be linked without altering features:

```powershell
.\.venv\Scripts\python.exe -m f1_ml_predictor register-collected-outcomes data/normalized/outcomes.parquet --sha256 EXPECTED_SHA256
```

Preview and install the recurring task with process-local execution policy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/install-collector-task.ps1 -Preview
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/install-collector-task.ps1
```

The original task was disabled on 2026-09-29 because its interactive PowerShell
process caused terminal popups. The replacement has been previewed, and its
windowless runner completed one network tick with a local JSONL log. It remains
uninstalled pending the local Windows credential. Use the manual command above
until its scheduled network tick is verified.

## Evaluation gate

`build-gold-rolling` verifies the frozen Gold benchmark, registry and exact
audited outcome file hashes. It creates a separate versioned benchmark with
same-season contiguous prior-race windows of 3, 5 and 10 rounds. A missing race
or any label published after the target cutoff makes the whole window missing.
Driver absences do not create results. The original feature snapshots and Gold
benchmark remain frozen. The rolling benchmark is keyed by its source Gold hash.

```powershell
.\.venv\Scripts\python.exe -m f1_ml_predictor build-gold-rolling
.\.venv\Scripts\python.exe -m f1_ml_predictor backtest --tier Gold --benchmark-dir data/benchmarks/gold_core_rolling_v1/SOURCE_GOLD_SHA256
.\.venv\Scripts\python.exe -m f1_ml_predictor compare-models --tier Gold --benchmark-dir data/benchmarks/gold_core_rolling_v1/SOURCE_GOLD_SHA256 --device auto
```

On the original 25-race Gold cohort, rolling finish means are present for 299,
219 and 60 driver-race observations at 3, 5 and 10 races respectively. Rolling
DNF rates are missing for all 506 observations because no DNF target has an
audited taxonomy category. The versioned model inputs include the rolling
values, missingness flags and audited history counts. The paired test cohort
contains 22 distinct races after earlier fit and calibration history.

The current bounded 2026 catalog in `docs/HISTORICAL_2026_CANDIDATES.json`
audited six additional candidates. Azerbaijan, Hungary, Belgium and Great
Britain passed; Italy and the Netherlands require review of later documents.
At that bounded audit checkpoint, Gold included 29 races, 594 driver-race
observations and 24 unique drivers. The five-season expansion above supersedes
that checkpoint for Gold Core coverage.
The separate `binary-dnf-v1` outcome catalog has 547 known labels, including 67
DNFs, with 47 unknown or excluded binary states. It retains exact OpenF1 and
Jolpica response hashes alongside the FIA final sources. The current rolling
comparison has 26 paired test races, 534 driver rows, 494 known DNF labels in
those test rows and no selected model. See [DNF audit](DNF_AUDIT_STATUS.md) and
[regression audit](BASELINE_REGRESSION_AUDIT.md).

```powershell
.\.venv\Scripts\python.exe scripts/prepare-2026-gold-candidates.py
.\.venv\Scripts\python.exe -m f1_ml_predictor audit-winter docs/HISTORICAL_2026_CANDIDATES.json --minimum-races 29 --reuse-retained
.\.venv\Scripts\python.exe scripts/collect-binary-dnf-evidence.py
.\.venv\Scripts\python.exe scripts/build-binary-dnf-benchmark.py
.\.venv\Scripts\python.exe -m f1_ml_predictor build-gold-rolling --benchmark-dir data/benchmarks/gold_core_binary_dnf_v1/4a809a21c40d5f65c86d1bd6bc420471119a33578228fa597379c0725e604a89/benchmark --catalog data/benchmarks/gold_core_binary_dnf_v1/4a809a21c40d5f65c86d1bd6bc420471119a33578228fa597379c0725e604a89/catalog.json
```

The [frozen protocol](EVALUATION_PROTOCOL_V1.md) requires 15 distinct paired Gold
test races for preliminary comparison and 25 for provisional selection, with
the existing baseline regression checks. Multiple timestamps for one event do
not increase this count. Development and Silver support engineering diagnostics
only. All metrics report tier and sample count. No champion or championship claim
is made from fixtures or an empty Gold benchmark.

```powershell
.\.venv\Scripts\python.exe -m f1_ml_predictor backtest --tier Gold --benchmark-dir data/benchmarks/gold_core
.\.venv\Scripts\python.exe -m f1_ml_predictor compare-models --tier Gold --benchmark-dir data/benchmarks/gold_core --device auto
```

Sigmoid, isotonic and identity DNF calibration use earlier held-out races. The
`--calibration-events` option can reserve multiple earlier events. Fit labels must
be available at the earliest calibration cutoff, and no outer test labels enter
calibration. Isotonic needs at least 100 known labels, ten per class and three
distinct scores; otherwise it skips with counts and a reason. Final composed race
marginals retain the existing reliability analysis and chronological evaluation.
GPU use retains the existing measured-benefit policy and CPU fallback.

## First Gold Core evaluation

The initial audit certified eight post-qualifying events: 2025 China, Sao Paulo,
Las Vegas, Qatar and Abu Dhabi; 2026 Australia, China and Japan. The benchmark has
166 driver-race observations, with unique drivers counted separately. Its Gold Parquet SHA256 is
`e016e7e2adeaa12a2754697d8d148ae5db562230058b822ed55eee5c66e07764`.
Entered drivers without a qualifying time retain null features backed by exact
roster, qualifying and event-context proofs. An empty row without those proofs
cannot claim Gold.

The standalone baseline run evaluated six races and wrote 126 prediction rows.
The stronger-model comparison reserved earlier calibration history and evaluated
the same five held-out events across all four installed backends and paired
baselines: 106 driver rows per backend, 424 total predictions. Winner metrics use
five races, podium metrics 106 drivers, finishing-position metrics 89 known
ordinals, and DNF metrics zero labels. All fits used CPU under the benefit policy.
DNF training/calibration report insufficient data; joint simulations retain the
explicit unvalidated Beta(1,1) prior, so these outputs cannot validate DNF quality.

The comparison reports, reliability bins, ranking scores, regression checks,
folds, source hashes and model artifacts remain under ignored local model/data
paths. Selection is deferred: there are only five independent paired test races,
missing DNF evaluation, and documented baseline regressions. No backend is named
champion and no WDC/WCC forecast is validated. Further evidence and prospective
confirmation remain necessary; the initial historical audit does not expand just
to improve these scores.

Archived forecasts stay omitted until exact operational run bytes and historical
release evidence are available. Open-Meteo [Single Runs documentation](https://open-meteo.com/en/docs/single-runs-api)
distinguishes initialization from release. Hindcasts and stitched historical
weather do not establish a past race-time forecast.
