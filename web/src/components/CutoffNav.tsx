import type { Cutoff, CutoffEntry } from "../data/schema";
import { cutoffLabel, formatDateTime, isShownCutoff } from "../format";

interface CutoffNavProps {
  cutoffs: CutoffEntry[];
  active: Cutoff;
  onSelect: (cutoff: Cutoff) => void;
}

/** Session-style snapshot navigation. Missing cutoffs are shown disabled, never invented. */
export function CutoffNav({ cutoffs, active, onSelect }: CutoffNavProps) {
  return (
    <nav className="cutoff-nav" aria-label="Prediction snapshots">
      <ol>
        {cutoffs.filter(isShownCutoff).map((entry, index) => {
          const current = entry.cutoff === active;
          return (
            <li key={entry.cutoff}>
              <button
                type="button"
                className={current ? "cutoff current" : "cutoff"}
                aria-current={current ? "true" : undefined}
                disabled={!entry.available}
                onClick={() => onSelect(entry.cutoff)}
                title={
                  entry.prediction_timestamp
                    ? `Cutoff ${formatDateTime(entry.prediction_timestamp)}`
                    : "No snapshot was published at this cutoff"
                }
              >
                <span className="cutoff-step">{String(index + 1).padStart(2, "0")}</span>
                <span className="cutoff-name">{cutoffLabel(entry.cutoff)}</span>
                <span className="cutoff-state">
                  {entry.available ? (current ? "Viewing" : "Available") : "Not published"}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
