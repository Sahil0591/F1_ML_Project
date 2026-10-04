import { useNavigate } from "react-router-dom";
import type { ExportIndex } from "../data/schema";

interface RaceNavigatorProps {
  index: ExportIndex;
  season: number;
  round: number;
}

/** Season and race selectors with previous and next race among exported races. */
export function RaceNavigator({ index, season, round }: RaceNavigatorProps) {
  const navigate = useNavigate();
  const seasonEntry = index.seasons.find((item) => item.season === season);
  const races = seasonEntry?.races ?? [];
  const all = index.seasons.flatMap((item) => item.races);
  const position = all.findIndex((race) => race.season === season && race.round === round);
  const previous = position > 0 ? all[position - 1] : undefined;
  const next = position >= 0 && position < all.length - 1 ? all[position + 1] : undefined;
  const go = (targetSeason: number, targetRound: number) =>
    navigate(`/predictions/${targetSeason}/${targetRound}`);

  return (
    <div className="race-nav" aria-label="Race navigation" role="navigation">
      <div className="race-nav-inner">
        <label className="select">
          <span>Season</span>
          <select
            value={season}
            onChange={(event) => {
              const chosen = index.seasons.find((item) => item.season === Number(event.target.value));
              const first = chosen?.races[chosen.races.length - 1];
              if (chosen && first) go(chosen.season, first.round);
            }}
          >
            {index.seasons.map((item) => (
              <option key={item.season} value={item.season}>
                {item.season}
              </option>
            ))}
          </select>
        </label>
        <label className="select select-race">
          <span>Race</span>
          <select value={round} onChange={(event) => go(season, Number(event.target.value))}>
            {races.map((race) => (
              <option key={race.round} value={race.round}>
                R{race.round} {race.race_name}
              </option>
            ))}
          </select>
        </label>
        <div className="race-nav-steps">
          <button
            type="button"
            className="step"
            disabled={!previous}
            onClick={() => previous && go(previous.season, previous.round)}
            aria-label={previous ? `Previous race: ${previous.race_name}` : "No previous race"}
          >
            <span aria-hidden="true">&larr;</span> Prev
          </button>
          <button
            type="button"
            className="step"
            disabled={!next}
            onClick={() => next && go(next.season, next.round)}
            aria-label={next ? `Next race: ${next.race_name}` : "No next race"}
          >
            Next <span aria-hidden="true">&rarr;</span>
          </button>
        </div>
      </div>
    </div>
  );
}
