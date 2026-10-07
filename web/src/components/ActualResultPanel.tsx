import type { Snapshot } from "../data/schema";
import { cutoffLabel, exact, formatPosition, formatProbability, formatSigned } from "../format";
import { DriverName, ProbabilityCell, TeamName } from "./Cells";
import { EmptyState } from "./States";

function Delta({ value }: { value: number | null }) {
  if (value === null) return <td className="num muted">n/a</td>;
  const tone = value > 0 ? "delta-up" : value < 0 ? "delta-down" : "delta-even";
  const words =
    value > 0
      ? `${value} places better than predicted`
      : value < 0
        ? `${-value} places worse than predicted`
        : "as predicted";
  return (
    <td className={`num ${tone}`} title={words}>
      <span aria-hidden="true">{value > 0 ? "▲ " : value < 0 ? "▼ " : "= "}</span>
      {formatSigned(value)}
      <span className="visually-hidden"> ({words})</span>
    </td>
  );
}

export function ActualResultPanel({ snapshot }: { snapshot: Snapshot }) {
  const result = snapshot.actual_result;
  if (result === null) {
    return (
      <EmptyState title="Actual result not yet available">
        <p>
          The {snapshot.identity.session === "sprint" ? "sprint" : "race"} classification
          appears here once the project&rsquo;s Jolpica ingestion has
          published it and the export is rerun. Predictions are never edited after the fact.
        </p>
      </EmptyState>
    );
  }
  const { summary } = result;
  const byId = new Map(result.classification.map((row) => [row.driver_id, row]));
  const name = (id: string | null) => (id ? (byId.get(id)?.driver_name ?? id) : "n/a");
  const before = snapshot.identity.session === "sprint" ? "Pre-sprint" : "Pre-race";
  return (
    <section aria-labelledby="result-title">
      <div className="section-head">
        <h2 id="result-title">Actual result against the prediction</h2>
        <p className="section-lede">
          Compared with the {cutoffLabel(snapshot.identity.cutoff).toLowerCase()} snapshot. Delta
          is predicted position minus finishing position, so positive means the driver finished
          ahead of the predicted order.
        </p>
      </div>
      <dl className="summary-grid">
        <div>
          <dt>Winner</dt>
          <dd>
            {name(summary.winner_id)}
            {summary.winner_probability !== null ? (
              <span className="summary-sub" title={exact(summary.winner_probability)}>
                {" "}
                {before.toLowerCase()} win chance {formatProbability(summary.winner_probability, snapshot.race.draws)}
              </span>
            ) : null}
          </dd>
        </div>
        <div>
          <dt>Predicted P1</dt>
          <dd>
            {name(summary.predicted_winner_id)}
            {summary.predicted_winner_id !== null ? (
              <span className="summary-sub">
                {summary.predicted_winner_id === summary.winner_id ? " won" : " did not win"}
              </span>
            ) : null}
          </dd>
        </div>
        <div>
          <dt>Podium chances given</dt>
          <dd>
            {summary.podium.length
              ? summary.podium
                  .map(
                    (item) =>
                      `${name(item.driver_id)} ${
                        item.podium_probability === null
                          ? "n/a"
                          : formatProbability(item.podium_probability, snapshot.race.draws)
                      }`,
                  )
                  .join(", ")
              : "n/a"}
          </dd>
        </div>
        <div>
          <dt>Retirements, DNF chance given</dt>
          <dd>
            {summary.retirements.length
              ? summary.retirements
                  .map(
                    (item) =>
                      `${name(item.driver_id)} ${
                        item.dnf_model_probability === null
                          ? "n/a"
                          : formatProbability(item.dnf_model_probability)
                      }`,
                  )
                  .join(", ")
              : "None"}
          </dd>
        </div>
        <div>
          <dt>Predicted order MAE</dt>
          <dd>
            {summary.predicted_position_mae === null
              ? "n/a"
              : `${formatPosition(summary.predicted_position_mae)} places`}
            <span className="summary-sub">
              {" "}
              {summary.exact_position_hits} of {summary.classified_finishers} exact
            </span>
          </dd>
        </div>
        <div>
          <dt>Clean expected MAE</dt>
          <dd>
            {summary.clean_expected_mae === null
              ? "n/a"
              : `${formatPosition(summary.clean_expected_mae)} places`}
          </dd>
        </div>
      </dl>
      <div className="table-scroll" tabIndex={0} role="region" aria-labelledby="result-title">
        <table className="results-table">
          <caption className="visually-hidden">
            Classification with {before.toLowerCase()} predictions
          </caption>
          <thead>
            <tr>
              <th scope="col" className="col-pos sticky-col">
                Pos.
              </th>
              <th scope="col" className="num">
                Pred.
              </th>
              <th scope="col" className="num">
                Delta
              </th>
              <th scope="col" className="sticky-col-2">
                Driver
              </th>
              <th scope="col">Team</th>
              <th scope="col">Result</th>
              <th scope="col" className="num">
                {before} Win Chance
              </th>
              <th scope="col" className="num">
                {before} Podium Chance
              </th>
            </tr>
          </thead>
          <tbody>
            {result.classification.map((row) => (
              <tr
                key={row.driver_id}
                className={row.classified && row.position !== null && row.position <= 3 ? `row-podium row-p${row.position}` : ""}
                data-testid="result-row"
              >
                <th scope="row" className="col-pos sticky-col">
                  <span className="pos">{row.classified ? row.position : (row.position_text ?? "NC")}</span>
                </th>
                <td className="num">{row.predicted_position !== null ? `P${row.predicted_position}` : "n/a"}</td>
                <Delta value={row.position_delta} />
                <td className="sticky-col-2">
                  <DriverName name={row.driver_name} code={row.driver_code} />
                </td>
                <td>
                  <TeamName id={row.constructor_id} name={row.constructor_name} />
                </td>
                <td>{row.status ?? "n/a"}</td>
                <ProbabilityCell value={row.winner_probability} draws={snapshot.race.draws} />
                <ProbabilityCell value={row.podium_probability} draws={snapshot.race.draws} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        Source: <code>{result.source_path}</code>
        {result.source_sha256 ? (
          <>
            {" "}
            (<code>{result.source_sha256.slice(0, 12)}</code>)
          </>
        ) : null}
        .
      </p>
    </section>
  );
}
