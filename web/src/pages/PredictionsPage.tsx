import { Link } from "react-router-dom";
import type { ExportIndex } from "../data/schema";
import { cutoffLabel, formatDate } from "../format";

export function PredictionsPage({ index }: { index: ExportIndex }) {
  return (
    <main id="main" className="page">
      <section className="hero hero-compact" aria-labelledby="predictions-title">
        <div className="hero-inner">
          <p className="hero-kicker">Archive</p>
          <h1 id="predictions-title">Race predictions</h1>
          <p className="hero-sub">
            Every exported development snapshot, newest season first. Snapshots are immutable;
            reruns at the same cutoff are listed as superseded.
          </p>
        </div>
      </section>
      {[...index.seasons].reverse().map((season) => (
        <section key={season.season} aria-labelledby={`season-${season.season}`} className="block">
          <div className="section-head section-head-row">
            <h2 id={`season-${season.season}`}>{season.season}</h2>
            <Link className="text-link" to={`/championship/${season.season}`}>
              Championship forecast
            </Link>
          </div>
          <div className="table-scroll" tabIndex={0} role="region" aria-labelledby={`season-${season.season}`}>
            <table className="results-table">
              <thead>
                <tr>
                  <th scope="col" className="col-pos">
                    Rnd.
                  </th>
                  <th scope="col">Grand Prix</th>
                  <th scope="col">Circuit</th>
                  <th scope="col">Date</th>
                  <th scope="col">Snapshots</th>
                  <th scope="col">Result</th>
                </tr>
              </thead>
              <tbody>
                {[...season.races].reverse().map((race) => (
                  <tr key={race.round}>
                    <th scope="row" className="col-pos">
                      {race.round}
                    </th>
                    <td>
                      <Link to={`/predictions/${race.season}/${race.round}`} className="race-link">
                        {race.race_name}
                      </Link>
                    </td>
                    <td>
                      {race.circuit_name}
                      {race.locality ? <span className="muted">, {race.locality}</span> : null}
                    </td>
                    <td>{formatDate(race.race_start)}</td>
                    <td>
                      <ul className="chips">
                        {race.cutoffs.map((entry) => (
                          <li
                            key={entry.cutoff}
                            className={entry.available ? "chip" : "chip chip-off"}
                            aria-label={`${cutoffLabel(entry.cutoff)} ${entry.available ? "available" : "not published"}`}
                          >
                            {cutoffLabel(entry.cutoff)}
                          </li>
                        ))}
                      </ul>
                    </td>
                    <td>{race.actual_result_available ? "Available" : "Not yet"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
      {index.excluded_runs.length ? (
        <details className="details block">
          <summary>Runs not exported ({index.excluded_runs.length})</summary>
          <ul className="plain-list">
            {index.excluded_runs.map((run) => (
              <li key={run.path}>
                <code>{run.path}</code>: {run.reason}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </main>
  );
}
