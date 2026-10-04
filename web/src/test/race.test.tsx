import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { App } from "../App";
import { ActualResultPanel } from "../components/ActualResultPanel";
import { ChampionshipTable } from "../components/ChampionshipTable";
import { Badges } from "../components/EventHero";
import { PredictedOrderTable, RaceProbabilitiesTable } from "../components/RaceTables";
import { RacePage } from "../pages/RacePage";
import { DRIVER_COUNT, makeIndex, makeSnapshot, stubFetch } from "./fixtures";

describe("predicted order", () => {
  it("renders 22 drivers as P1 to P22 with no duplicates", () => {
    render(<PredictedOrderTable snapshot={makeSnapshot()} />);
    const rows = screen.getAllByTestId("order-row");
    expect(rows).toHaveLength(DRIVER_COUNT);
    const positions = rows.map((row) => within(row).getAllByRole("rowheader")[0]?.textContent);
    expect(positions).toEqual(Array.from({ length: DRIVER_COUNT }, (_, i) => `P${i + 1}`));
    expect(new Set(positions).size).toBe(DRIVER_COUNT);
  });

  it("shows clean expected position from the artifact unchanged", () => {
    const snapshot = makeSnapshot();
    render(<PredictedOrderTable snapshot={snapshot} />);
    const first = screen.getAllByTestId("order-row")[0];
    if (!first) throw new Error("no rows");
    const source = snapshot.race.drivers.find((driver) => driver.predicted_position === 1);
    const cell = within(first).getByText("1.25");
    expect(cell).toHaveAttribute("data-value", String(source?.clean_expected_position));
  });

  it("does not invent an order for snapshots that predate it", () => {
    render(<PredictedOrderTable snapshot={makeSnapshot({ withOrder: false })} />);
    expect(screen.getByText("Predicted order not in this snapshot")).toBeInTheDocument();
    expect(screen.queryAllByTestId("order-row")).toHaveLength(0);
  });
});

describe("race probabilities", () => {
  it("lists every driver with win, podium, DNF and finish values", () => {
    render(<RaceProbabilitiesTable snapshot={makeSnapshot()} />);
    expect(screen.getAllByTestId("prob-row")).toHaveLength(DRIVER_COUNT);
    expect(screen.getByRole("columnheader", { name: "Clean Race" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "80% Range" })).toBeInTheDocument();
    expect(screen.getByText(/Clean Race removes retirements/)).toBeInTheDocument();
  });

  it("keeps small and zero probabilities visible instead of 0.0%", () => {
    render(<RaceProbabilitiesTable snapshot={makeSnapshot()} />);
    expect(screen.getAllByText("0 of 65,536").length).toBeGreaterThan(0);
    expect(screen.queryByText("0.0%")).not.toBeInTheDocument();
    const small = screen.getAllByText(/^\d\.\d+e?-?\d*%$|^0\.0\d+%$/);
    expect(small.length).toBeGreaterThan(0);
  });
});

describe("championship tables", () => {
  it("renders every WDC entry with source values in data attributes", () => {
    const snapshot = makeSnapshot();
    render(<ChampionshipTable kind="wdc" championship={snapshot.championship} />);
    const rows = screen.getAllByTestId("wdc-row");
    expect(rows).toHaveLength(snapshot.championship.wdc.entries.length);
    const leader = snapshot.championship.wdc.entries[0];
    expect(within(rows[0] as HTMLElement).getByText("77%").closest("td")).toHaveAttribute(
      "data-value",
      String(leader?.title_probability),
    );
    expect(screen.getByRole("heading", { name: "Projected Drivers Championship" })).toBeVisible();
  });

  it("renders every constructor in the WCC table", () => {
    const snapshot = makeSnapshot();
    render(<ChampionshipTable kind="wcc" championship={snapshot.championship} />);
    expect(screen.getAllByTestId("wcc-row")).toHaveLength(snapshot.championship.wcc.entries.length);
    expect(screen.getByRole("columnheader", { name: "WCC Chance" })).toBeInTheDocument();
    expect(screen.getAllByText(">99%").length).toBeGreaterThan(0);
  });
});

describe("actual result", () => {
  it("says the result is not available before the race", () => {
    render(<ActualResultPanel snapshot={makeSnapshot()} />);
    expect(screen.getByText("Actual result not yet available")).toBeInTheDocument();
  });

  it("compares the classification with the prediction", () => {
    render(<ActualResultPanel snapshot={makeSnapshot({ withResult: true })} />);
    const rows = screen.getAllByTestId("result-row");
    expect(rows).toHaveLength(DRIVER_COUNT);
    const winner = rows[0] as HTMLElement;
    expect(within(winner).getByText("P2")).toBeInTheDocument();
    expect(within(winner).getByText(/1 places better than predicted/)).toBeInTheDocument();
    expect(within(rows[DRIVER_COUNT - 1] as HTMLElement).getByText("R")).toBeInTheDocument();
    expect(screen.getByText("Predicted P1")).toBeInTheDocument();
    expect(screen.getByText(/did not win/)).toBeInTheDocument();
  });
});

describe("status badges", () => {
  it("shows development only and OOD badges", () => {
    render(<Badges snapshot={makeSnapshot()} />);
    expect(screen.getByText("Development only")).toBeInTheDocument();
    expect(screen.getByText("Out of distribution")).toBeInTheDocument();
    expect(screen.getByText("Unseen circuit")).toBeInTheDocument();
  });

  it("omits the OOD badge for in-distribution runs", () => {
    render(<Badges snapshot={makeSnapshot({ ood: false })} />);
    expect(screen.queryByText("Out of distribution")).not.toBeInTheDocument();
  });
});

function renderRace(path = "/predictions/2026/16") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/predictions/:season/:round" element={<RacePage index={makeIndex()} />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("race page", () => {
  it("switches between cutoff snapshots and disables unpublished ones", async () => {
    stubFetch({
      "2026/round-16/pre_weekend.json": makeSnapshot({ cutoff: "pre_weekend" }),
      "2026/round-16/post_qualifying.json": makeSnapshot({ cutoff: "post_qualifying", winnerShift: 1 }),
    });
    renderRace();
    expect(await screen.findByRole("heading", { name: "Fixture Grand Prix" })).toBeInTheDocument();
    const nav = screen.getByRole("navigation", { name: "Prediction snapshots" });
    const current = within(nav).getByRole("button", { name: /Post-qualifying/ });
    expect(current).toHaveAttribute("aria-current", "true");
    expect(within(nav).queryByRole("button", { name: /Post-practice/ })).not.toBeInTheDocument();
    expect(within(nav).getByRole("button", { name: /Pre-race/ })).toBeDisabled();
    await userEvent.click(within(nav).getByRole("button", { name: /Pre-weekend/ }));
    const updated = await screen.findByRole("button", { name: /Pre-weekend.*Viewing/ });
    expect(updated).toHaveAttribute("aria-current", "true");
    expect(screen.getByRole("button", { name: /Post-qualifying.*Available/ })).not.toBeDisabled();
  });

  it("supports keyboard navigation between tabs", async () => {
    stubFetch({
      "2026/round-16/post_qualifying.json": makeSnapshot({ cutoff: "post_qualifying" }),
    });
    renderRace();
    const tab = await screen.findByRole("tab", { name: "Predicted Order" });
    tab.focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Race Probabilities" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await userEvent.keyboard("{End}");
    expect(screen.getByRole("tab", { name: "Model Details" })).toHaveFocus();
    expect(screen.getByRole("heading", { name: "Model details" })).toBeInTheDocument();
  });
});

describe("app shell", () => {
  it("opens and closes the mobile navigation menu", async () => {
    stubFetch({
      "index.json": makeIndex(),
      "2026/round-16/post_qualifying.json": makeSnapshot({ cutoff: "post_qualifying" }),
    });
    render(
      <MemoryRouter initialEntries={["/predictions"]}>
        <App />
      </MemoryRouter>,
    );
    const toggle = await screen.findByRole("button", { name: "Open menu" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(toggle);
    expect(screen.getByRole("button", { name: "Close menu" })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(screen.getByRole("navigation", { name: "Primary" })).toHaveClass("open");
  });

  it("shows a validation error for malformed JSON", async () => {
    stubFetch({ "index.json": "{ not json" });
    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    );
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveAttribute("data-error-kind", "malformed");
    expect(within(alert).getByText("Prediction file failed validation")).toBeInTheDocument();
  });

  it("rejects documents that do not match the schema", async () => {
    const broken = { ...makeIndex(), seasons: [{ season: "2026" }] };
    stubFetch({ "index.json": broken });
    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    );
    expect(await screen.findByRole("alert")).toHaveAttribute("data-error-kind", "malformed");
  });

  it("reports an unsupported export schema version", async () => {
    stubFetch({ "index.json": { ...makeIndex(), schema_version: 0 } });
    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    );
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveAttribute("data-error-kind", "unsupported");
  });

  it("reports a missing artifact", async () => {
    stubFetch({});
    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    );
    expect(await screen.findByRole("alert")).toHaveAttribute("data-error-kind", "missing");
  });
});
