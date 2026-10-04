/** Neutral identification stripes per constructor; not team branding or logos. */
const TEAM_COLOURS: Record<string, string> = {
  mercedes: "#27a39a",
  ferrari: "#c8102e",
  mclaren: "#f28c28",
  red_bull: "#2b3f8c",
  rb: "#5b7fd6",
  alpine: "#3a8fd6",
  aston_martin: "#1d6b55",
  williams: "#3f6fd8",
  haas: "#8a8d93",
  audi: "#6b6f76",
  sauber: "#3fa34d",
  cadillac: "#3b3b45",
};

export function teamColour(constructorId: string | null | undefined): string {
  return (constructorId && TEAM_COLOURS[constructorId]) || "#9a9aa2";
}
