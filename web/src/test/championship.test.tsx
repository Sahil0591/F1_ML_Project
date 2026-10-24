import { render, screen, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { ChampionshipPage } from "../pages/ChampionshipPage";
import { makeIndex, makeSeason, makeSnapshot, stubFetch } from "./fixtures";

function renderChampionship() {
  return render(
    <MemoryRouter initialEntries={["/championship/2026"]}>
      <Routes>
        <Route path="/championship/:season" element={<ChampionshipPage index={makeIndex()} />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("championship page", () => {
  it("shows WDC and WCC projections with equal prominence", async () => {
    stubFetch({
      "2026/season.json": makeSeason(),
      "2026/round-16/pre_weekend.json": makeSnapshot(),
    });
    renderChampionship();
    const wdc = await screen.findByRole("heading", { name: "Projected Drivers Championship" });
    const wcc = screen.getByRole("heading", { name: "Projected Constructors Championship" });
    expect(wdc.tagName).toBe(wcc.tagName);
    expect(screen.getAllByTestId("wcc-row")).toHaveLength(11);
    expect(screen.getByRole("heading", { name: "Season in progress" })).toBeInTheDocument();
  });

  it("orders drivers by points now and keeps the projected position", async () => {
    const snapshot = makeSnapshot();
    // The projected leader D01 sits second on points now.
    snapshot.championship.wdc.entries = snapshot.championship.wdc.entries.map((entry, index) =>
      index < 2 ? { ...entry, points_now: index === 0 ? 290 : 310 } : entry,
    );
    stubFetch({
      "2026/season.json": makeSeason(),
      "2026/round-16/pre_weekend.json": snapshot,
    });
    renderChampionship();
    const rows = await screen.findAllByTestId("wdc-row");
    expect(within(rows[0] as HTMLElement).getByText("D02")).toBeInTheDocument();
    // Proj. and Most Likely Finish both keep D02's projection.
    expect(within(rows[0] as HTMLElement).getAllByText("P2")).toHaveLength(2);
    expect(within(rows[1] as HTMLElement).getByText("D01")).toBeInTheDocument();
  });

  it("compares final standings with the last projection once the season is complete", async () => {
    stubFetch({
      "2026/season.json": makeSeason(true),
      "2026/round-16/pre_weekend.json": makeSnapshot(),
    });
    renderChampionship();
    expect(await screen.findByRole("heading", { name: "Final Drivers Championship" })).toBeVisible();
    expect(screen.getAllByTestId("final-driver-row")).toHaveLength(2);
    expect(screen.getAllByTestId("final-constructor-row")).toHaveLength(1);
    expect(screen.getByText("+42")).toBeInTheDocument();
  });

  it("explains a missing championship season", () => {
    render(
      <MemoryRouter initialEntries={["/championship/1999"]}>
        <Routes>
          <Route path="/championship/:season" element={<ChampionshipPage index={makeIndex()} />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByText("No championship data for this season")).toBeInTheDocument();
  });
});
