import type { Championship, StandingEntry } from "../data/schema";
import { exact, formatCoarse, formatDate, formatPoints } from "../format";
import { teamColour } from "../teams";
import { DriverName, ProbabilityCell, TeamName } from "./Cells";

export type TitleKind = "wdc" | "wcc";

const TITLES: Record<TitleKind, { heading: string; chance: string; entity: string }> = {
  wdc: { heading: "Projected Drivers Championship", chance: "Title Chance", entity: "Driver" },
  wcc: {
    heading: "Projected Constructors Championship",
    chance: "WCC Chance",
    entity: "Constructor",
  },
};

function PointsCell({ value }: { value: number }) {
  return (
    <td className="num points-cell" title={exact(value)} data-value={exact(value)}>
      {formatPoints(value)}
    </td>
  );
}

function Contenders({ entries, kind }: { entries: StandingEntry[]; kind: TitleKind }) {
  const contenders = entries
    .filter((entry) => entry.title_probability >= 0.01)
    .sort((a, b) => b.title_probability - a.title_probability);
  if (!contenders.length) return null;
  return (
    <ul className="contenders" aria-label={`${TITLES[kind].chance} for realistic contenders`}>
      {contenders.map((entry) => (
        <li key={entry.id}>
          <span className="contender-name">{entry.name}</span>
          <span className="contender-bar" aria-hidden="true">
            <span
              style={{
                width: `${entry.title_probability * 100}%`,
                background: teamColour(kind === "wdc" ? entry.constructor_id : entry.id),
              }}
            />
          </span>
          <span className="contender-value" title={exact(entry.title_probability)}>
            {formatCoarse(entry.title_probability)}
          </span>
        </li>
      ))}
    </ul>
  );
}

interface ChampionshipTableProps {
  kind: TitleKind;
  championship: Championship;
  headingLevel?: 2 | 3;
}

export function ChampionshipTable({ kind, championship, headingLevel = 2 }: ChampionshipTableProps) {
  const table = championship[kind];
  const copy = TITLES[kind];
  // Drivers are listed in standings order; equal points fall back to the projection.
  const entries = [...table.entries].sort((a, b) =>
    kind === "wdc"
      ? b.points_now - a.points_now || a.projected_position - b.projected_position
      : a.projected_position - b.projected_position,
  );
  const Heading = headingLevel === 2 ? "h2" : "h3";
  const headingId = `${kind}-title`;
  return (
    <section
      id={`${kind}-standings`}
      aria-labelledby={headingId}
      className="championship"
      data-testid={`${kind}-section`}
    >
      <div className="section-head">
        <Heading id={headingId}>{copy.heading}</Heading>
        <p className="section-lede">
          {championship.simulations.toLocaleString("en-GB")} simulated seasons over{" "}
          {championship.worlds.toLocaleString("en-GB")} model-uncertainty worlds. Points now are
          the published standings as of {formatDate(championship.standings_available_at)}.{" "}
          {kind === "wdc"
            ? "Drivers are ordered by points now; Proj. is the projected order by mean final points."
            : "Pos. is the projected order by mean final points."}{" "}
          Chances are shown to whole percentage points;
          exact values are in each cell&rsquo;s tooltip.
        </p>
      </div>
      <Contenders entries={entries} kind={kind} />
      <div className="table-scroll" tabIndex={0} role="region" aria-labelledby={headingId}>
        <table className="results-table standings-table">
          <caption className="visually-hidden">{copy.heading}</caption>
          <thead>
            <tr>
              <th scope="col" className="col-pos sticky-col">
                Pos.
              </th>
              <th scope="col" className="sticky-col-2">
                {copy.entity}
              </th>
              {kind === "wdc" ? <th scope="col">Team</th> : null}
              <th scope="col" className="num">
                Points Now
              </th>
              {kind === "wdc" ? (
                <th scope="col" className="num">
                  Proj.
                </th>
              ) : null}
              <th scope="col" className="num">
                Expected Final
              </th>
              <th scope="col" className="num">
                {copy.chance}
              </th>
              <th scope="col" className="num">
                Top 3
              </th>
              <th scope="col" className="num">
                Most Likely Finish
              </th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry, index) => {
              const position = kind === "wdc" ? index + 1 : entry.projected_position;
              return (
                <tr
                  key={entry.id}
                  className={position <= 3 ? `row-podium row-p${position}` : ""}
                  data-testid={`${kind}-row`}
                >
                  <th scope="row" className="col-pos sticky-col">
                    <span className="pos">{position}</span>
                  </th>
                  <td className="sticky-col-2">
                    {kind === "wdc" ? (
                      <DriverName name={entry.name} code={entry.code} />
                    ) : (
                      <TeamName id={entry.id} name={entry.name} />
                    )}
                  </td>
                  {kind === "wdc" ? (
                    <td>
                      {entry.constructor_name ? (
                        <TeamName id={entry.constructor_id} name={entry.constructor_name} />
                      ) : (
                        "n/a"
                      )}
                    </td>
                  ) : null}
                  <PointsCell value={entry.points_now} />
                  {kind === "wdc" ? <td className="num">P{entry.projected_position}</td> : null}
                  <td
                    className="num projected"
                    title={exact(entry.expected_final_points)}
                    data-value={exact(entry.expected_final_points)}
                  >
                    {Math.round(entry.expected_final_points).toLocaleString("en-GB")}
                  </td>
                  <ProbabilityCell value={entry.title_probability} coarse />
                  <ProbabilityCell value={entry.top3_probability} coarse />
                  <td className="num">P{entry.most_likely_position}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="note">
        Unresolved title tie probability: {formatCoarse(table.unresolved_tie_probability)}. Ties
        stay unresolved because historical FIA countback is not available to the simulator.
      </p>
    </section>
  );
}
