# Weather forecast addendum (2026-11-06)

Gold weather values were all missing because Open-Meteo's archive records a
run's initialisation, not its release. This addendum closes that gap with a
release margin and tests whether race-time forecasts improve post-qualifying
predictions. They do not, so live predictions are unchanged.

## Evidence

`scripts/audit-historical-weather.py <coverage report>` freezes one ECMWF IFS
HRES run per Gold race from the Open-Meteo
[Single Runs API](https://open-meteo.com/en/docs/single-runs-api):

- The run is the latest 00Z or 12Z initialisation at least 12 hours before the
  post-qualifying cutoff. Open-Meteo documents global model output as public
  4-6 hours after initialisation. The 06Z and 18Z runs are not archived for the
  whole period, so they are not used.
- The archive starts on 14 March 2024. Earlier races keep weather missing.
- Race starts come from the coverage report's audited direct catalog. The
  Jolpica schedule places the 2024 Las Vegas race a day early.
- Response bytes are retained under `data/raw/open_meteo_runs/objects`. Every
  value is replayed from them when Gold history loads.

The audit `weather-forecast-audit-ec1b1b0c...` covers all 99 Gold races: 63
audited and 36 before the archive. Rain is measured over the first two race hours
and over the six hours before the start, since a soaked track stays wet. The
first version counted only the race hours and called Brazil 2024 dry: its forecast
had rain all morning (2.8 mm in the six hours before the start) easing just
before the start. With both windows, 18 races had at least 0.5 mm forecast,
against 8 before; some, such as Miami 2024, were probably dry by the start.

## Contract

`post_qualifying_weather` is addendum `weather-addendum-v2` to protocol v3. It
uses the post-qualifying cutoff and predictors and adds
`forecast_race_precipitation_mm`, `forecast_pre_race_precipitation_mm`,
`forecast_temperature_2m` and `forecast_wind_speed_10m`, on the forecast's
release clock. The DNF model also
gets them. Run it with
`evaluate-cutoffs --contracts post_qualifying post_qualifying_weather`.

## Result

Paired against `post_qualifying` on 96 outer races (weather minus base, 95%
race bootstrap):

| Model | Races | Winner log loss | Podium Brier | Position MAE |
| --- | ---: | ---: | ---: | ---: |
| ensemble_all | 96 | -0.002 [-0.012, +0.006] | +0.0001 [-0.0002, +0.0003] | +0.002 [-0.001, +0.007] |
| ensemble_all | 18 wet | -0.002 [-0.044, +0.030] | +0.0002 [-0.0006, +0.0009] | +0.004 [-0.006, +0.016] |
| selected_pipeline | 96 | +0.003 [-0.006, +0.013] | -0.0001 [-0.0004, +0.0003] | +0.002 [-0.004, +0.007] |

Logistic DNF Brier is worse with weather (+0.020 [+0.001, +0.047]). The race-only
first version was worse on its 8 wet races (ensemble winner log loss +0.040).

Race-level weather is the same for every driver, so it cannot say who gains in
the wet. Against the official out-of-fold expectations, drivers differ in the
wet: over the 18 wet races Verstappen finished 1.3 places further ahead of the
expectation than in dry races (95% bootstrap +0.16 to +2.57), as did Norris
(+1.7) and Hulkenberg (+2.0), while Russell and Leclerc leaned the other way.
Nineteen drivers were compared, so one such interval is expected by chance. A
per-driver wet skill applied when rain is forecast is the next test.

Live predictions keep the frozen `post_qualifying` and `pre_race` contracts. Re-run
the weather audit after each new Gold race; a new audit is needed whenever the
Gold race set changes.
