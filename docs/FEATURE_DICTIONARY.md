# Pre-Race Feature Dictionary

Feature version: `1`. Snapshot: after qualifying and before race start.

## Input and availability contract

The builder consumes `FeatureInputs`: a published event specification, roster and
qualifying versions, and optional result/session/forecast/context versions.
`PublishedTable` requires a UTC availability timestamp and nonempty evidence
reference for known publications. Unknown optional versions are excluded; unknown
or late required versions cause an error. Existing row availability cannot be
backdated by the wrapper.

Publication evidence is an explicit curator trust boundary. A timestamp or evidence
string cannot prove historical release by itself. Verify the exact source version
before constructing a known publication. The builder never invents historical
availability for retrospective API downloads. Qualifying cancellation and
pit-lane-start encoding are not yet supported.

The latest known complete version is selected at the cutoff. Equal-time ambiguity
is rejected. Prior results must precede the target event identifier and have
completed before cutoff. Current/future rounds are excluded even with misdated
source timestamps. Each row records the conservative maximum `feature_timestamp`,
input references/content hashes, event/completion timestamps, window, source policy,
and prediction cutoff. No target labels enter this table.

## Values

Every numeric feature has a Boolean `<feature>_missing` column. Missing values are
null, never filled from future results or actual race weather.

| Feature | Definition and allowed inputs |
| --- | --- |
| `qualifying_position` | Published current qualifying classification position. |
| `qualifying_last_session_seconds` | Latest available stage in Q3, Q2, Q1 order. Not minimum time across stages. |
| `grid_position` | Published roster/grid position only. No current race-result grid or qualifying fallback. |
| `teammate_qualifying_position_delta` | Driver position minus mean known position of current roster teammates. |
| `recent_finish_mean` | Mean classified position in the driver's last `form_window` known prior starts, default five, possibly across seasons. |
| `recent_dnf_rate` | Mean explicit audited Boolean DNF labels in those starts. Unknown statuses stay unlabeled. |
| `constructor_recent_finish_mean` | Mean classified position of constructor drivers in the last `form_window` known prior events. Performance proxy, not physical pace. |
| `recent_pit_stop_seconds` | Mean supplied prior-race mean pit duration across recent starts. Duration definition must be consistent upstream. |
| `practice_best_seconds` | Minimum best lap among known practice driver/compound summaries from one source. |
| `practice_median_seconds` | Unweighted mean summary median across known practice sessions/compounds. Conditions and fuel loads are not corrected. |
| `tyre_age_mean` | Mean known median tyre age across those summaries. Not estimated degradation. |
| `tyre_compound_count` | Distinct known practice compounds. All unknown yields missing, not zero. |
| `forecast_temperature_2m` | Celsius at first hourly forecast target in `[race_start, race_start + 1 hour)`. |
| `forecast_precipitation_probability` | Precipitation probability at that target, percent, bounded 0-100. |
| `forecast_wind_speed_10m` | Wind speed at that target, metres per second. |
| `driver_championship_points` | Published same-season points at a common round before the target round. Sprint points only if supplied upstream. |
| `constructor_championship_points` | Constructor points from that standings publication and current roster identity. |
| `circuit_length_km` | Published static length for the selected circuit. |
| `is_street_circuit` | Published Boolean represented as 0.0 or 1.0. |

`history_count` counts selected prior driver starts. `session_source` identifies
FastF1 or OpenF1 when practice features exist. Do not average their summaries: their
quality filters differ. Detailed telemetry and tyre-degradation models remain
deferred until value and availability justify them.

## Manifest and CLI

Run `python -m f1_ml_predictor build-snapshot <manifest.json> --root <workspace>`.
JSON requires `version: 1`, `prediction_timestamp`, an `event` object, arrays `rosters`
and `qualifying`, and optional `history`, `sessions`, `forecasts`, `standings`, `circuit`.
Each publication declares its kind, exact file SHA-256, workspace-relative Parquet
path, availability, and evidence reference. Paths cannot escape the root. Files are
hash-verified before reading, so corrections cannot reuse an old file's evidence.

Publication shape (placeholders are not historical evidence):

```json
{
  "kind": "qualifying",
  "path": "data/normalized/verified/qualifying.parquet",
  "sha256": "REPLACE_WITH_EXACT_FILE_SHA256",
  "available_at": null,
  "evidence_reference": "REPLACE_WITH_VERIFIED_SOURCE_VERSION_REFERENCE"
}
```

Kinds: `roster`, `qualifying`, `race_results`, `session_summary`, `forecast`, `standings`,
`circuit`. Event fields: `season`, `round`, `circuit_id`, `race_start`,
`qualifying_completed_at`, `available_at`, `evidence_reference`. History items have
`season`, `round`, `race_completed_at`, and `publication`. Timestamps are ISO UTC
strings. Outputs are immutable content-addressed Parquet under `data/features`.

## Validation Gate

Offline fixtures demonstrate calculations and leakage controls, not predictive
performance. Real rolling backtests require verified historical snapshots or an
accumulated prospective archive. Fixture tests support no accuracy or calibration
claim.
