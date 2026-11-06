import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ShadowPanel } from "../components/ShadowPanel";
import type { ShadowVariant, Snapshot } from "../data/schema";
import { snapshotSchema } from "../data/schema";
import { DRIVER_COUNT, makeSnapshot } from "./fixtures";

function variant(
  snapshot: Snapshot,
  id: string,
  label: string,
  shift: number,
  extra: Partial<ShadowVariant> = {},
): ShadowVariant {
  return {
    id,
    label,
    description: `${label} description`,
    backtest: id === "official" ? null : `${label} backtest`,
    status: "published",
    reason: null,
    applied: null,
    drivers: snapshot.race.drivers.map((driver, index) => ({
      driver_id: driver.driver_id,
      winner_probability: index === 0 ? Math.min(driver.winner_probability + shift, 1) : driver.winner_probability,
      podium_probability: driver.podium_probability,
      expected_position: driver.expected_position,
    })),
    scores: null,
    ...extra,
  };
}

function withShadows(rehearsal: string | null = null): Snapshot {
  const snapshot = makeSnapshot({ cutoff: "post_qualifying" });
  return {
    ...snapshot,
    shadows: {
      source_path: "data/predictions/development/shadows/x.json",
      source_sha256: "a".repeat(64),
      created_at: "2026-11-07T19:00:00+00:00",
      rehearsal,
      forecast: {
        status: "audited",
        model: "ecmwf_ifs",
        run_initialized_at: "2026-11-07T00:00:00+00:00",
        available_at: "2026-11-07T12:00:00+00:00",
        rain_before_mm: 2.4,
        rain_during_mm: 1.1,
        temperature_c: 16.2,
        wind_ms: 3.1,
        wet: true,
      },
      variants: [
        variant(snapshot, "official", "Official", 0),
        variant(snapshot, "two_stage", "Two-stage", 0.1),
        variant(snapshot, "wet_skill", "Wet skill", 0, { applied: true }),
        {
          ...variant(snapshot, "strategy", "Strategy", 0),
          status: "skipped",
          reason: "needs audits",
          drivers: null,
        },
      ],
    },
  };
}

describe("shadow predictions", () => {
  it("explains when a snapshot has none", () => {
    render(<ShadowPanel snapshot={makeSnapshot()} />);
    expect(screen.getByText("No shadow predictions for this snapshot")).toBeInTheDocument();
  });

  it("shows the forecast, every version and a row per driver", () => {
    render(<ShadowPanel snapshot={withShadows()} />);
    expect(screen.getByText("2.4 mm")).toBeInTheDocument();
    expect(screen.getByText("Wet adjustment on")).toBeInTheDocument();
    expect(screen.getByText("Skipped: needs audits")).toBeInTheDocument();
    expect(screen.getAllByTestId("shadow-variant")).toHaveLength(4);
    expect(screen.getAllByTestId("shadow-row")).toHaveLength(DRIVER_COUNT);
    expect(screen.queryByText(/Rehearsal\./)).not.toBeInTheDocument();
  });

  it("marks shadow values against the official chance", () => {
    render(<ShadowPanel snapshot={withShadows()} />);
    const gaps = document.querySelectorAll(".shadow-gap");
    expect(gaps.length).toBe(1);
    expect(gaps[0]?.textContent).toContain("+10.0");
  });

  it("labels a rehearsal as one", () => {
    render(<ShadowPanel snapshot={withShadows("Rebuilt after the race.")} />);
    expect(screen.getByRole("note")).toHaveTextContent("Rehearsal. Rebuilt after the race.");
  });

  it("does not render shadows that failed verification", () => {
    render(<ShadowPanel snapshot={{ ...makeSnapshot(), shadows: { invalid_reason: "roster differs" } }} />);
    expect(screen.getByText("Shadow file failed verification")).toBeInTheDocument();
    expect(screen.queryAllByTestId("shadow-row")).toHaveLength(0);
  });

  it("parses exports written before shadows existed", () => {
    const older: Partial<Snapshot> = { ...makeSnapshot() };
    delete older.shadows;
    expect(snapshotSchema.parse(older).shadows).toBeNull();
  });

  it("scores each version once the result is in", () => {
    const snapshot = withShadows();
    if (snapshot.shadows === null || "invalid_reason" in snapshot.shadows) throw new Error("fixture");
    const scores = {
      winner_id: snapshot.race.drivers[0]?.driver_id ?? "",
      winner_probability: 0.4,
      winner_log_loss: 0.92,
      favourite_won: true,
      podium_hits: 2,
      position_mae: 2.5,
    };
    snapshot.shadows.variants = snapshot.shadows.variants.map((item) =>
      item.drivers ? { ...item, scores } : item,
    );
    render(<ShadowPanel snapshot={snapshot} />);
    const rows = screen.getAllByTestId("shadow-score");
    expect(rows).toHaveLength(3);
    expect(within(rows[0] as HTMLElement).getByText("2 of 3")).toBeInTheDocument();
  });
});
