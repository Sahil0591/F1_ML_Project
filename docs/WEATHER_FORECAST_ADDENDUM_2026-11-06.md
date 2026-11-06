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

The audit `weather-forecast-audit-ba9063c6...` covers all 99 Gold races: 63
audited and 36 before the archive. Eight races had at least 0.5 mm forecast
over the first two race hours, among them Canada and Britain 2024, Australia and
Belgium 2025, and 2026 rounds 4 and 16.

## Contract

`post_qualifying_weather` is addendum `weather-addendum-v1` to protocol v3. It
uses the post-qualifying cutoff and predictors and adds
`forecast_race_precipitation_mm`, `forecast_temperature_2m` and
`forecast_wind_speed_10m`, on the forecast's release clock. The DNF model also
gets them. Run it with
`evaluate-cutoffs --contracts post_qualifying post_qualifying_weather`.

## Result

Paired against `post_qualifying` on 96 outer races (weather minus base, 95%
race bootstrap):

| Model | Races | Winner log loss | Podium Brier | Position MAE |
| --- | ---: | ---: | ---: | ---: |
| ensemble_all | 96 | -0.001 [-0.009, +0.007] | +0.0001 [-0.0002, +0.0003] | +0.003 [-0.001, +0.007] |
| ensemble_all | 8 wet | +0.040 [+0.003, +0.078] | +0.0013 [+0.0002, +0.0025] | -0.004 [-0.018, +0.008] |
| selected_pipeline | 96 | +0.003 [-0.008, +0.013] | -0.0001 [-0.0005, +0.0003] | +0.001 [-0.004, +0.006] |

Logistic DNF Brier is worse with weather (+0.019 [+0.001, +0.046]).

Forecast-wet races were not less predictable either. Widening the race-order
temperature by a factor of 1.4 to 2.8 on those races raised their mean
calibration objective from 1.98 to between 2.24 and 3.10, so a rain-variance
rule is not supported. Favourites won most of them.

Live predictions keep the frozen `post_qualifying` and `pre_race` contracts. Re-run
the weather audit after each new Gold race; a new audit is needed whenever the
Gold race set changes.
