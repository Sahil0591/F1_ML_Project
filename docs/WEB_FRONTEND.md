# Web frontend

The `web/` app is a Vite, React and TypeScript results page for the development
predictions. It shows the P1 to P22 predicted order, race probabilities, the
projected Drivers and Constructors Championships, actual results once they are
ingested, and model details. It never computes predictions: every number comes
from a published run through the Python export adapter.

## Artifact flow

```text
predict-next-race  ->  data/predictions/development/next_race/season=S/round=R/<cutoff>/<run id>/
                         manifest.json, race_distribution.json, championship.json,
                         predictions.parquet (immutable, gitignored)
ingest-season S    ->  data/normalized/season=S/round=RR/results.parquet (actual results)
export-web         ->  web/public/data/index.json
                       web/public/data/S/round-RR/<cutoff>.json
                       web/public/data/S/season.json
npm run build      ->  web/dist/ (static site, data copied from public/data)
```

`python -m f1_ml_predictor export-web` reads the run directories and never
writes to them. For each run it:

1. checks every artifact hash recorded in `manifest.json` and that the
   championship is bound to the run's race distribution;
2. requires `methodology` `cutoff-specific-v3`, `development_only` and
   `validated_forecast: false`; legacy c52b674 runs are listed as excluded;
3. validates the race contract (unique drivers, probabilities in [0, 1], finish
   distributions summing to 1 and agreeing with win and podium, a permutation
   for `predicted_position`, Parquet rows equal to `race_distribution.json`) and
   the title contract (complete entries, position distributions summing to 1);
4. verifies the schedule observation the run used, for circuit location;
5. writes typed JSON with sorted keys. The output is built in a temporary
   directory and swapped in, so a failed export leaves the previous one intact.
   The exporter refuses to replace a directory it did not write.

When several runs exist for one race and cutoff, the most recently created run
is shown and the others are listed as superseded. Cutoffs with no run are shown
disabled; nothing is filled in.

## Display contract

Values copied unchanged: `predicted_position`, `clean_expected_position`,
`clean_most_likely_position`, `expected_position`, `most_likely_position`,
`winner_probability`, `podium_probability`, `dnf_model_probability`,
`position_interval_80`, title probabilities, mean final points, final position
distributions, starting points and fixed-strength sensitivity.

Derived in the exporter, not in React:

- championship Top 3 is the sum of the first three final position probabilities;
- most likely final position is the mode of the final position distribution;
- projected position orders entries by mean final points, then title probability,
  as the run report does;
- actual result comparison: delta is predicted position minus finishing
  position, clean error is finishing position minus clean expected position, and
  MAE uses classified finishers only;
- `generated_after_race_start` flags runs created after the race started, such as
  a replay at an earlier cutoff.

Race probabilities keep small values visible (`0.0046%`, or `0 of 65,536` when
the event never occurred in the draws). Championship probabilities use whole
percentage points (`<1%`, `>99%`) as the simulator documents. Every rounded cell
carries the exact source value in its tooltip and `data-value` attribute.

`predicted_position` is a modelled order by clean-race expectation with win
probability as the tie break. The page says it is not an FIA classification.

Season results appear once every round of a season has audited points in the
scoring ledger; final standings are compared with the last projection.

## Commands

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m f1_ml_predictor export-web
cd web
npm install
npm run dev        # http://localhost:5173
npm run typecheck
npm run lint
npm run test
npm run build      # static output in web/dist
```

`export-web --output <dir>` writes elsewhere. Exported JSON is gitignored like
the run artifacts it comes from; rerun the export after each prediction run or
result ingestion.

## Routes

| Route | Content |
| --- | --- |
| `/` | Redirects to the latest exported race |
| `/predictions` | Every exported race with its cutoff snapshots |
| `/predictions/:season/:round` | Event page; `?cutoff=` and `?tab=` deep link a snapshot and tab |
| `/championship/:season` | WDC and WCC projections, forecast history and season results |

## Deployment

`npm run build` produces a static site. Serve `web/dist` from any static host
with a fallback to `index.html` for client routes. Set `F1_WEB_BASE=/subpath/`
at build time when the site is not served from the domain root. Export the data
before building, because `public/data` is copied into `dist/data`.

## Data states

The app shows a skeleton while loading and separate states for a missing file,
a file that fails schema validation, an unsupported export schema version, a race
with no actual result yet, a season with no final results, and a snapshot that
predates the predicted order (65c35d6); the last is never reconstructed in the
browser.

Branding is original. No Formula 1 logos, fonts, CSS or imagery are used; team
colours are plain identification stripes.
