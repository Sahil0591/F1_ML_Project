import { Link, useParams } from "react-router-dom";
import { ChampionshipTable } from "../components/ChampionshipTable";
import { DriverName, ProbabilityCell, TeamName } from "../components/Cells";
import { LineChart } from "../components/LineChart";
import { EmptyState, ErrorState, LoadingState } from "../components/States";
import { loadSeason, loadSnapshot } from "../data/api";
import type { ExportIndex, Season, SeasonComparisonRow, Snapshot } from "../data/schema";
import { useResource } from "../data/useResource";
import { cutoffLabel, exact, formatDateTime, formatPoints, formatSigned } from "../format";

function Evolution({ season, snapshot }: { season: Season; snapshot: Snapshot }) {
  if (season.snapshots.length < 2) {
    return (
      <p className="note">
        Title probability evolution appears once two or more prediction snapshots exist for{" "}
        {season.season}. Currently {season.snapshots.length} snapshot
        {season.snapshots.length === 1 ? "" : "s"}.
      </p>
    );
  }
  const labels = season.snapshots.map((item) => `R${item.round} ${cutoffLabel(item.cutoff)}`);
  const top = (kind: "wdc" | "wcc") =>
    [...snapshot.championship[kind].entries]
      .sort((a, b) => b.title_probability - a.title_probability)
      .slice(0, 4);
  return (
    <div className="chart-pair">
      <LineChart
        title="Drivers title probability by snapshot"
        xLabels={labels}
        series={top("wdc").map((entry) => ({
          id: entry.id,
          label: entry.code ?? entry.name,
          colourKey: entry.constructor_id,
          values: season.snapshots.map((item) => item.wdc_title[entry.id] ?? null),
        }))}
      />
      <LineChart
        title="Constructors title probability by snapshot"
        xLabels={labels}
        series={top("wcc").map((entry) => ({
          id: entry.id,
          label: entry.name,
          colourKey: entry.id,
          values: season.snapshots.map((item) => item.wcc_title[entry.id] ?? null),
        }))}
      />
    </div>
  );
}

function FinalTable({
  title,
  rows,
  entity,
}: {
  title: string;
  rows: SeasonComparisonRow[];
  entity: "driver" | "constructor";
}) {
  const id = `final-${entity}`;
  return (
    <section aria-labelledby={id}>
      <h3 id={id}>{title}</h3>
      <div className="table-scroll" tabIndex={0} role="region" aria-labelledby={id}>
        <table className="results-table">
          <thead>
            <tr>
              <th scope="col" className="col-pos sticky-col">
                Pos.
              </th>
              <th scope="col" className="sticky-col-2">
                {entity === "driver" ? "Driver" : "Constructor"}
              </th>
              <th scope="col" className="num">
                Actual Points
              </th>
              <th scope="col" className="num">
                Last Projected Pos.
              </th>
              <th scope="col" className="num">
                Projected Points
              </th>
              <th scope="col" className="num">
                Title Chance
              </th>
              <th scope="col" className="num">
                Points Error
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} data-testid={`final-${entity}-row`}>
                <th scope="row" className="col-pos sticky-col">
                  <span className="pos">{row.final_position}</span>
                </th>
                <td className="sticky-col-2">
                  {entity === "driver" ? (
                    <DriverName name={row.name} code={null} />
                  ) : (
                    <TeamName id={row.id} name={row.name} />
                  )}
                </td>
                <td className="num">{formatPoints(row.actual_points)}</td>
                <td className="num">
                  {row.projected_position === null ? "n/a" : row.projected_position}
                </td>
                <td className="num">
                  {row.projected_final_points === null
                    ? "n/a"
                    : Math.round(row.projected_final_points).toLocaleString("en-GB")}
                </td>
                <ProbabilityCell value={row.title_probability} coarse />
                <td
                  className="num"
                  title={row.points_error === null ? undefined : exact(row.points_error)}
                >
                  {row.points_error === null ? "n/a" : formatSigned(row.points_error)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SeasonResults({ season }: { season: Season }) {
  if (!season.completed || !season.final_standings || !season.final_comparison) {
    return (
      <EmptyState title="Season in progress">
        <p>
          Final WDC and WCC results appear here once every round of {season.season} has audited
          points in the scoring ledger. The comparison then uses the last snapshot before the
          season ended.
        </p>
      </EmptyState>
    );
  }
  return (
    <>
      <p className="section-lede">
        Final standings from the {season.final_standings.source}, compared with the last
        projection (round {season.latest.round}, {cutoffLabel(season.latest.cutoff).toLowerCase()}).
        Points error is actual minus projected.
      </p>
      <FinalTable title="Final Drivers Championship" rows={season.final_comparison.drivers} entity="driver" />
      <FinalTable
        title="Final Constructors Championship"
        rows={season.final_comparison.constructors}
        entity="constructor"
      />
    </>
  );
}

export function ChampionshipPage({ index }: { index: ExportIndex }) {
  const params = useParams();
  const seasonNumber = Number(params.season);
  const entry = index.seasons.find((item) => item.season === seasonNumber);
  const season = useResource(entry ? entry.season_path : null, () =>
    loadSeason(entry?.season_path ?? ""),
  );
  const latestPath = season.status === "ready" ? season.data.latest.path : null;
  const snapshot = useResource(latestPath, () => loadSnapshot(latestPath ?? ""));

  if (!entry) {
    return (
      <main id="main" className="page">
        <EmptyState title="No championship data for this season">
          <p>No WDC or WCC forecast has been exported for {params.season}.</p>
        </EmptyState>
      </main>
    );
  }
  if (season.status === "loading" || (season.status === "ready" && snapshot.status === "loading")) {
    return (
      <main id="main" className="page">
        <LoadingState label="Loading championship forecast" />
      </main>
    );
  }
  if (season.status === "error") {
    return (
      <main id="main" className="page">
        <ErrorState error={season.error} />
      </main>
    );
  }
  if (snapshot.status === "error") {
    return (
      <main id="main" className="page">
        <ErrorState error={snapshot.error} />
      </main>
    );
  }
  if (snapshot.status !== "ready") return null;
  const data = season.data;
  const latest = snapshot.data;
  return (
    <main id="main" className="page">
      <section
        className="hero hero-compact hero-championship"
        aria-labelledby="season-title"
        data-season={data.season}
      >
        <div className="hero-inner">
          <p className="hero-kicker">
            Season {data.season} <span aria-hidden="true">/</span> {data.total_rounds} rounds
          </p>
          <h1 id="season-title">{data.season} Championship forecast</h1>
          <p className="hero-sub">
            Latest projection from{" "}
            <Link to={`/predictions/${data.season}/${data.latest.round}?cutoff=${data.latest.cutoff}`}>
              round {data.latest.round}, {cutoffLabel(data.latest.cutoff).toLowerCase()}
            </Link>{" "}
            (cutoff {formatDateTime(data.latest.prediction_timestamp)}).
          </p>
          <ul className="badges" aria-label="Forecast status">
            {latest.identity.development_only ? (
              <li className="badge badge-dev">Development only</li>
            ) : null}
            <li className="badge badge-cutoff">{data.completed ? "Season complete" : "Season in progress"}</li>
          </ul>
          <p className="hero-warning">{latest.warning}</p>
        </div>
      </section>
      <nav className="results-nav" aria-label="Championship sections">
        <span className="results-nav-label">Season view</span>
        <a href="#wdc-standings">Drivers</a>
        <a href="#wcc-standings">Constructors</a>
        <a href="#evolution-title">Forecast history</a>
        <a href="#season-results-title">Season results</a>
      </nav>
      <div className="title-grid">
        <ChampionshipTable kind="wdc" championship={latest.championship} />
        <ChampionshipTable kind="wcc" championship={latest.championship} />
      </div>
      <section aria-labelledby="evolution-title" className="block">
        <h2 id="evolution-title">Forecast history</h2>
        <Evolution season={data} snapshot={latest} />
        <div className="table-scroll" tabIndex={0} role="region" aria-labelledby="evolution-title">
          <table className="results-table compact">
            <thead>
              <tr>
                <th scope="col">Snapshot</th>
                <th scope="col">Cutoff time</th>
                <th scope="col">WDC leader</th>
                <th scope="col" className="num">
                  Chance
                </th>
                <th scope="col">WCC leader</th>
                <th scope="col" className="num">
                  Chance
                </th>
              </tr>
            </thead>
            <tbody>
              {data.snapshots.map((item) => {
                const wdc = Object.entries(item.wdc_title).sort((a, b) => b[1] - a[1])[0];
                const wcc = Object.entries(item.wcc_title).sort((a, b) => b[1] - a[1])[0];
                const nameOf = (kind: "wdc" | "wcc", id: string) =>
                  latest.championship[kind].entries.find((e) => e.id === id)?.name ?? id;
                return (
                  <tr key={item.path}>
                    <th scope="row">
                      <Link to={`/predictions/${data.season}/${item.round}?cutoff=${item.cutoff}`}>
                        R{item.round} {cutoffLabel(item.cutoff)}
                      </Link>
                    </th>
                    <td>{formatDateTime(item.prediction_timestamp)}</td>
                    <td>{wdc ? nameOf("wdc", wdc[0]) : "n/a"}</td>
                    <ProbabilityCell value={wdc ? wdc[1] : null} coarse />
                    <td>{wcc ? nameOf("wcc", wcc[0]) : "n/a"}</td>
                    <ProbabilityCell value={wcc ? wcc[1] : null} coarse />
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
      <section aria-labelledby="season-results-title" className="block">
        <h2 id="season-results-title">Season results</h2>
        <SeasonResults season={data} />
      </section>
    </main>
  );
}
