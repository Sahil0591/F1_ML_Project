import type { RaceDriver, Snapshot } from "../data/schema";
import { formatProbability } from "../format";
import { DriverName, NumberCell, ProbabilityCell, TeamName } from "./Cells";
import { EmptyState } from "./States";

function positionClass(position: number | null): string {
  return position !== null && position <= 3 ? `row-podium row-p${position}` : "";
}

export function PredictedOrderTable({ snapshot }: { snapshot: Snapshot }) {
  const { race } = snapshot;
  if (!race.predicted_order_available) {
    return (
      <EmptyState title="Predicted order not in this snapshot">
        <p>
          This run predates the P1 to P{race.drivers.length} predicted order, which the pipeline
          added in commit 65c35d6. The app does not reconstruct it. Win, podium and finish
          probabilities for this run are under Race Probabilities.
        </p>
      </EmptyState>
    );
  }
  const drivers = [...race.drivers].sort(
    (a, b) => (a.predicted_position ?? 0) - (b.predicted_position ?? 0),
  );
  return (
    <section aria-labelledby="order-title">
      <div className="section-head">
        <h2 id="order-title">Predicted finishing order</h2>
        <p className="section-lede">
          P1 to P{drivers.length} ranked by clean-race expected position, with win probability as
          the tie break. This is a modelled order, not an FIA classification or a deterministic
          prediction: the win and podium columns say how likely each place really is.
        </p>
      </div>
      <div className="table-scroll" tabIndex={0} role="region" aria-labelledby="order-title">
        <table className="results-table">
          <caption className="visually-hidden">
            Predicted finishing order, {drivers.length} drivers
          </caption>
          <thead>
            <tr>
              <th scope="col" className="col-pos sticky-col">
                Pred.
              </th>
              <th scope="col" className="sticky-col-2">
                Driver
              </th>
              <th scope="col">Team</th>
              <th scope="col" className="num">
                Clean Expected
              </th>
              <th scope="col" className="num">
                Most Likely
              </th>
              <th scope="col" className="num">
                Expected incl. DNF
              </th>
              <th scope="col" className="num">
                Win
              </th>
              <th scope="col" className="num">
                Podium
              </th>
            </tr>
          </thead>
          <tbody>
            {drivers.map((driver) => (
              <tr
                key={driver.driver_id}
                className={positionClass(driver.predicted_position)}
                data-testid="order-row"
              >
                <th scope="row" className="col-pos sticky-col">
                  <span className="pos">P{driver.predicted_position}</span>
                </th>
                <td className="sticky-col-2">
                  <DriverName name={driver.driver_name} code={driver.driver_code} />
                </td>
                <td>
                  <TeamName id={driver.constructor_id} name={driver.constructor_name} />
                </td>
                <NumberCell value={driver.clean_expected_position} />
                <td className="num">
                  {driver.clean_most_likely_position !== null
                    ? `P${driver.clean_most_likely_position}`
                    : "n/a"}
                </td>
                <NumberCell value={driver.expected_position} />
                <ProbabilityCell value={driver.winner_probability} draws={race.draws} bar />
                <ProbabilityCell value={driver.podium_probability} draws={race.draws} bar />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <dl className="footnotes">
        <div>
          <dt>Clean Expected</dt>
          <dd>Expected finishing position with retirements switched off in the same draws.</dd>
        </div>
        <div>
          <dt>Most Likely</dt>
          <dd>Single most frequent position in the clean-race draws.</dd>
        </div>
        <div>
          <dt>Expected incl. DNF</dt>
          <dd>Expected position over every draw, with retirements placed last.</dd>
        </div>
      </dl>
    </section>
  );
}

export function RaceProbabilitiesTable({ snapshot }: { snapshot: Snapshot }) {
  const { race } = snapshot;
  const drivers: RaceDriver[] = [...race.drivers].sort(
    (a, b) => b.winner_probability - a.winner_probability || a.expected_position - b.expected_position,
  );
  const fieldWide = race.dnf_field_wide ? drivers[0]?.dnf_model_probability : undefined;
  return (
    <section aria-labelledby="prob-title">
      <div className="section-head">
        <h2 id="prob-title">Race probabilities</h2>
        <p className="section-lede">
          From {race.draws.toLocaleString("en-GB")} joint simulated races. Expected includes
          retirement outcomes; Clean Race removes retirements from the same simulated draws. Small
          values keep their precision, and &ldquo;0 of {race.draws.toLocaleString("en-GB")}&rdquo;
          means the event never occurred in the draws.
        </p>
      </div>
      <div className="table-scroll" tabIndex={0} role="region" aria-labelledby="prob-title">
        <table className="results-table">
          <caption className="visually-hidden">Race probabilities by driver</caption>
          <thead>
            <tr>
              <th scope="col" className="sticky-col">
                Driver
              </th>
              <th scope="col">Team</th>
              <th scope="col" className="num">
                Win
              </th>
              <th scope="col" className="num">
                Podium
              </th>
              <th scope="col" className="num">
                DNF
              </th>
              <th scope="col" className="num">
                Expected Finish
              </th>
              <th scope="col" className="num">
                Clean Race
              </th>
              <th scope="col" className="num">
                80% Range
              </th>
            </tr>
          </thead>
          <tbody>
            {drivers.map((driver) => (
              <tr key={driver.driver_id} data-testid="prob-row">
                <th scope="row" className="sticky-col">
                  <DriverName name={driver.driver_name} code={driver.driver_code} />
                </th>
                <td>
                  <TeamName id={driver.constructor_id} name={driver.constructor_name} />
                </td>
                <ProbabilityCell value={driver.winner_probability} draws={race.draws} bar />
                <ProbabilityCell value={driver.podium_probability} draws={race.draws} bar />
                <ProbabilityCell value={driver.dnf_model_probability} />
                <NumberCell value={driver.expected_position} />
                <NumberCell value={driver.clean_expected_position} />
                <td className="num">
                  P{driver.position_interval_80[0]} to P{driver.position_interval_80[1]}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {fieldWide !== undefined ? (
        <p className="note">
          <strong>DNF is field-wide.</strong> The selected DNF model (
          <code>{snapshot.model.dnf_model}</code>) gives every driver{" "}
          {formatProbability(fieldWide)}: driver-specific reliability features did not beat the
          audited base rate out of fold, so this is not an individual prediction.
        </p>
      ) : null}
    </section>
  );
}
