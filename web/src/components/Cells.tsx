import { exact, formatCoarse, formatProbability } from "../format";
import { teamColour } from "../teams";

interface ProbabilityCellProps {
  value: number | null;
  draws?: number;
  /** Show a proportional bar; the text value is always shown so colour is never the only cue. */
  bar?: boolean;
  coarse?: boolean;
}

export function ProbabilityCell({ value, draws, bar = false, coarse = false }: ProbabilityCellProps) {
  if (value === null) {
    return <td className="num muted">n/a</td>;
  }
  const text = coarse ? formatCoarse(value) : formatProbability(value, draws);
  return (
    <td className="num prob" title={exact(value)} data-value={exact(value)}>
      {bar ? (
        <span className="prob-bar" aria-hidden="true">
          <span style={{ width: `${Math.max(value * 100, value > 0 ? 1 : 0)}%` }} />
        </span>
      ) : null}
      <span className="prob-text">{text}</span>
    </td>
  );
}

export function NumberCell({ value, digits = 2 }: { value: number | null; digits?: number }) {
  if (value === null) return <td className="num muted">n/a</td>;
  return (
    <td className="num" title={exact(value)} data-value={exact(value)}>
      {value.toFixed(digits)}
    </td>
  );
}

export function TeamName({ id, name }: { id: string | null; name: string }) {
  return (
    <span className="team">
      <span className="team-stripe" style={{ background: teamColour(id) }} aria-hidden="true" />
      {name}
    </span>
  );
}

export function DriverName({ name, code }: { name: string; code: string | null }) {
  const parts = name.split(" ");
  const family = parts.length > 1 ? parts.slice(-1)[0] : name;
  const given = parts.length > 1 ? parts.slice(0, -1).join(" ") : "";
  return (
    <span className="driver">
      {given ? <span className="driver-given">{given}</span> : null}
      <span className="driver-family">{family}</span>
      {code ? <span className="driver-code">{code}</span> : null}
    </span>
  );
}
