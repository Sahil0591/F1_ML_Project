/**
 * Display formatting only. Values are never recomputed here; exact source values
 * are kept in `title` attributes next to every rounded number.
 */

const integer = new Intl.NumberFormat("en-GB");

/** Race probabilities: keep small nonzero values visible, as the backend report does. */
export function formatProbability(value: number, draws?: number): string {
  if (value === 0) return draws ? `0 of ${integer.format(draws)}` : "0";
  if (value < 0.001) return `${(100 * value).toPrecision(2)}%`;
  return `${(100 * value).toFixed(1)}%`;
}

/** Championship probabilities: whole percentage points, as the simulator documents. */
export function formatCoarse(value: number): string {
  if (value === 0) return "0 in sim.";
  if (value < 0.01) return "<1%";
  if (value > 0.99 && value < 1) return ">99%";
  return `${Math.round(100 * value)}%`;
}

export function formatPosition(value: number, digits = 2): string {
  return value.toFixed(digits);
}

export function formatPoints(value: number): string {
  return Number.isInteger(value) ? integer.format(value) : value.toFixed(1);
}

export function formatSigned(value: number, digits = 0): string {
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return digits ? (0).toFixed(digits) : "0";
  return `${value > 0 ? "+" : "−"}${text}`;
}

export function exact(value: number): string {
  return String(value);
}

const dateFormat = new Intl.DateTimeFormat("en-GB", {
  day: "2-digit",
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});
const dateTimeFormat = new Intl.DateTimeFormat("en-GB", {
  day: "2-digit",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  timeZone: "UTC",
  hourCycle: "h23",
});

export function formatDate(iso: string): string {
  return dateFormat.format(new Date(iso));
}

export function formatDateTime(iso: string): string {
  return `${dateTimeFormat.format(new Date(iso))} UTC`;
}

export const CUTOFF_LABELS: Record<string, string> = {
  pre_weekend: "Pre-weekend",
  post_practice: "Post-practice",
  post_qualifying: "Post-qualifying",
  pre_race: "Pre-race",
};

/**
 * Practice is not captured live, so post-practice runs are never produced.
 * Hide that cutoff unless a run for it actually exists.
 */
export function isShownCutoff(entry: { cutoff: string; available: boolean }): boolean {
  return entry.cutoff !== "post_practice" || entry.available;
}

export function cutoffLabel(cutoff: string): string {
  return CUTOFF_LABELS[cutoff] ?? cutoff.replace(/_/g, " ");
}

export function humanize(identifier: string): string {
  return identifier.replace(/_/g, " ");
}
