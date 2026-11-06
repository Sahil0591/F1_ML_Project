import type { ShadowVariant, Snapshot } from "../data/schema";
import { cutoffLabel, exact, formatPosition, formatProbability } from "../format";
import { DriverName, TeamName } from "./Cells";
import { EmptyState } from "./States";

function mm(value: number | null): string {
  return value === null ? "n/a" : `${value.toFixed(1)} mm`;
}

/** Shadow value with its gap to the official one, so the change reads without arithmetic. */
function ShadowCell({ value, official }: { value: number | null; official: number | null }) {
  if (value === null) return <td className="num muted">n/a</td>;
  const gap = official === null ? null : value - official;
  const points = gap === null ? 0 : Math.round(gap * 1000) / 10;
  const tone = points >= 1 ? "delta-up" : points <= -1 ? "delta-down" : "delta-even";
  return (
    <td className="num prob" title={exact(value)} data-value={exact(value)}>
      <span className="prob-text">{formatProbability(value)}</span>
      {gap !== null && Math.abs(points) >= 1 ? (
        <span className={`shadow-gap ${tone}`}>
          {points > 0 ? "+" : "−"}
          {Math.abs(points).toFixed(1)}
          <span className="visually-hidden"> points against official</span>
        </span>
      ) : null}
    </td>
  );
}

function VariantStatus({ variant }: { variant: ShadowVariant }) {
  if (variant.status === "skipped") return <span className="shadow-tag muted">Skipped</span>;
  if (variant.id === "wet_skill") {
    return variant.applied ? (
      <span className="shadow-tag shadow-tag-on">Wet adjustment on</span>
    ) : (
      <span className="shadow-tag">Dry forecast, adjustment off</span>
    );
  }
  return null;
}

export function ShadowPanel({ snapshot }: { snapshot: Snapshot }) {
  const shadows = snapshot.shadows;
  if (shadows === null) {
    return (
      <EmptyState title="No shadow predictions for this snapshot">
        <p>
          Shadow predictions are frozen next to post-qualifying and pre-race runs made from 7
          November 2026 onwards. Earlier snapshots and pre-weekend runs have none.
        </p>
      </EmptyState>
    );
  }
  if ("invalid_reason" in shadows) {
    return (
      <EmptyState title="Shadow file failed verification">
        <p>
          The shadow predictions for this run were not shown because they did not match the
          published run: {shadows.invalid_reason}. The official prediction is unaffected.
        </p>
      </EmptyState>
    );
  }
  const { forecast } = shadows;
  const official = shadows.variants.find((item) => item.id === "official");
  const published = shadows.variants.filter((item) => item.drivers !== null);
  const values = new Map(
    published.map((variant) => [
      variant.id,
      new Map((variant.drivers ?? []).map((row) => [row.driver_id, row])),
    ]),
  );
  const baseline = values.get("official");
  const order = [...snapshot.race.drivers].sort(
    (a, b) =>
      (baseline?.get(b.driver_id)?.winner_probability ?? 0) -
      (baseline?.get(a.driver_id)?.winner_probability ?? 0),
  );
  const names = new Map(snapshot.race.drivers.map((row) => [row.driver_id, row]));
  const scored = published.some((variant) => variant.scores !== null);
  return (
    <section aria-labelledby="shadow-title" className="shadow-panel">
      <div className="section-head">
        <h2 id="shadow-title">Shadow predictions</h2>
        <p className="section-lede">
          Not official. Candidate methods that did not pass the backtest gate, frozen with the{" "}
          {cutoffLabel(snapshot.identity.cutoff).toLowerCase()} prediction and scored once the race
          is run. They never change the published numbers.
        </p>
      </div>
      {shadows.rehearsal ? (
        <p className="shadow-banner" role="note">
          <strong>Rehearsal.</strong> {shadows.rehearsal}
        </p>
      ) : null}

      <h3 className="shadow-subhead">Forecast the shadows were given</h3>
      {forecast.status === "audited" ? (
        <dl className="summary-grid">
          <div>
            <dt>Rain in the 6 h before the start</dt>
            <dd>{mm(forecast.rain_before_mm)}</dd>
          </div>
          <div>
            <dt>Rain over the first 2 race hours</dt>
            <dd>{mm(forecast.rain_during_mm)}</dd>
          </div>
          <div>
            <dt>Temperature at the start</dt>
            <dd>{forecast.temperature_c === null ? "n/a" : `${forecast.temperature_c.toFixed(1)} °C`}</dd>
          </div>
          <div>
            <dt>Wind at the start</dt>
            <dd>{forecast.wind_ms === null ? "n/a" : `${forecast.wind_ms.toFixed(1)} m/s`}</dd>
          </div>
          <div>
            <dt>Wet call</dt>
            <dd>
              {forecast.wet === null ? "n/a" : forecast.wet ? "Wet" : "Dry"}
              <span className="summary-sub"> 0.5 mm or more before or during</span>
            </dd>
          </div>
          <div>
            <dt>Source</dt>
            <dd>
              ECMWF run {forecast.run_initialized_at?.slice(0, 16).replace("T", " ") ?? "n/a"} UTC
              <span className="summary-sub"> released before the cutoff</span>
            </dd>
          </div>
        </dl>
      ) : (
        <p className="note">No complete forecast was available ({forecast.status}).</p>
      )}

      <h3 className="shadow-subhead">The versions</h3>
      <ul className="shadow-variants">
        {shadows.variants.map((variant) => (
          <li key={variant.id} data-testid="shadow-variant">
            <span className="shadow-variant-name">{variant.label}</span> <VariantStatus variant={variant} />
            <span className="shadow-variant-text">{variant.description}</span>
            {variant.backtest ? <span className="shadow-variant-text muted">{variant.backtest}</span> : null}
            {variant.status === "skipped" && variant.reason ? (
              <span className="shadow-variant-text muted">Skipped: {variant.reason}</span>
            ) : null}
          </li>
        ))}
      </ul>

      <h3 className="shadow-subhead" id="shadow-table-title">
        Win and podium chances
      </h3>
      <p className="note">
        Small figures beside a shadow value are its difference from the official chance, in
        percentage points, shown from 1 point.
      </p>
      <div className="table-scroll" tabIndex={0} role="region" aria-labelledby="shadow-table-title">
        <table className="results-table shadow-table">
          <thead>
            <tr>
              <th scope="col" rowSpan={2} className="sticky-col">
                Driver
              </th>
              <th scope="col" rowSpan={2}>
                Team
              </th>
              {published.map((variant) => (
                <th scope="colgroup" colSpan={2} key={variant.id} className="num shadow-group">
                  {variant.label}
                </th>
              ))}
            </tr>
            <tr>
              {published.map((variant) => (
                <FragmentHeads key={variant.id} />
              ))}
            </tr>
          </thead>
          <tbody>
            {order.map((driver) => (
              <tr key={driver.driver_id} data-testid="shadow-row">
                <th scope="row" className="sticky-col">
                  <DriverName name={driver.driver_name} code={driver.driver_code} />
                </th>
                <td>
                  <TeamName id={driver.constructor_id} name={driver.constructor_name} />
                </td>
                {published.map((variant) => {
                  const row = values.get(variant.id)?.get(driver.driver_id);
                  const base = baseline?.get(driver.driver_id);
                  const isOfficial = variant.id === "official";
                  return [
                    <ShadowCell
                      key={`${variant.id}-win`}
                      value={row?.winner_probability ?? null}
                      official={isOfficial ? null : (base?.winner_probability ?? null)}
                    />,
                    <ShadowCell
                      key={`${variant.id}-podium`}
                      value={row?.podium_probability ?? null}
                      official={isOfficial ? null : (base?.podium_probability ?? null)}
                    />,
                  ];
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="shadow-subhead" id="shadow-score-title">
        How each version did
      </h3>
      {scored ? (
        <div className="table-scroll" tabIndex={0} role="region" aria-labelledby="shadow-score-title">
          <table className="results-table">
            <thead>
              <tr>
                <th scope="col">Version</th>
                <th scope="col" className="num">
                  Winner&rsquo;s chance
                </th>
                <th scope="col">Favourite won</th>
                <th scope="col" className="num">
                  Podium hits
                </th>
                <th scope="col" className="num">
                  Finishing MAE
                </th>
              </tr>
            </thead>
            <tbody>
              {published.map((variant) =>
                variant.scores === null ? null : (
                  <tr key={variant.id} data-testid="shadow-score">
                    <th scope="row">{variant.label}</th>
                    <td className="num" title={exact(variant.scores.winner_probability)}>
                      {formatProbability(variant.scores.winner_probability)}
                    </td>
                    <td>{variant.scores.favourite_won ? "Yes" : "No"}</td>
                    <td className="num">{variant.scores.podium_hits} of 3</td>
                    <td className="num">
                      {variant.scores.position_mae === null
                        ? "n/a"
                        : `${formatPosition(variant.scores.position_mae)} places`}
                    </td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
          <p className="note">
            Winner: {names.get(published.find((v) => v.scores)?.scores?.winner_id ?? "")?.driver_name ?? "n/a"}.
            One race decides nothing; the running comparison is in <code>scripts/score-shadows.py</code>.
          </p>
        </div>
      ) : (
        <p className="note">Scores appear once the race result is ingested and the export is rerun.</p>
      )}

      <p className="note">
        Source: <code>{shadows.source_path}</code> (<code>{shadows.source_sha256.slice(0, 12)}</code>).
        {official ? null : " No official column was recorded."}
      </p>
    </section>
  );
}

function FragmentHeads() {
  return (
    <>
      <th scope="col" className="num">
        Win
      </th>
      <th scope="col" className="num">
        Podium
      </th>
    </>
  );
}
