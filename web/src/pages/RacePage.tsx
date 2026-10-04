import { useParams, useSearchParams } from "react-router-dom";
import { ActualResultPanel } from "../components/ActualResultPanel";
import { ChampionshipTable } from "../components/ChampionshipTable";
import { CutoffNav } from "../components/CutoffNav";
import { EventHero } from "../components/EventHero";
import { LineChart } from "../components/LineChart";
import { ModelDetails } from "../components/ModelDetails";
import { RaceNavigator } from "../components/RaceNavigator";
import { PredictedOrderTable, RaceProbabilitiesTable } from "../components/RaceTables";
import { EmptyState, ErrorState, LoadingState } from "../components/States";
import { Tabs } from "../components/Tabs";
import { loadSnapshot } from "../data/api";
import type { Cutoff, ExportIndex, RaceEntry, Snapshot } from "../data/schema";
import { cutoffSchema } from "../data/schema";
import { useResource } from "../data/useResource";
import { cutoffLabel } from "../format";

const RACE_TABS = [
  { id: "order", label: "Predicted Order" },
  { id: "probabilities", label: "Race Probabilities" },
  { id: "wdc", label: "Drivers Championship" },
  { id: "wcc", label: "Constructors Championship" },
  { id: "result", label: "Actual Result" },
  { id: "model", label: "Model Details" },
] as const;

type TabId = (typeof RACE_TABS)[number]["id"];

function isTab(value: string | null): value is TabId {
  return RACE_TABS.some((tab) => tab.id === value);
}

function WeekendEvolution({ race, current }: { race: RaceEntry; current: Snapshot }) {
  const available = race.cutoffs.filter((entry) => entry.available && entry.path);
  const paths = available.map((entry) => entry.path ?? "");
  const key = available.length > 1 ? paths.join("|") : null;
  const snapshots = useResource(key, () => Promise.all(paths.map((path) => loadSnapshot(path))));
  if (available.length < 2) {
    return (
      <p className="note">
        Win probability through the weekend appears when more than one cutoff snapshot exists for
        this race.
      </p>
    );
  }
  if (snapshots.status !== "ready") return null;
  const leaders = [...current.race.drivers]
    .sort((a, b) => b.winner_probability - a.winner_probability)
    .slice(0, 5);
  return (
    <LineChart
      title="Win probability through the weekend, current leaders"
      xLabels={available.map((entry) => cutoffLabel(entry.cutoff))}
      series={leaders.map((driver) => ({
        id: driver.driver_id,
        label: driver.driver_code,
        colourKey: driver.constructor_id,
        values: snapshots.data.map(
          (snapshot) =>
            snapshot.race.drivers.find((item) => item.driver_id === driver.driver_id)
              ?.winner_probability ?? null,
        ),
      }))}
    />
  );
}

export function RacePage({ index }: { index: ExportIndex }) {
  const params = useParams();
  const [search, setSearch] = useSearchParams();
  const season = Number(params.season);
  const round = Number(params.round);
  const race = index.seasons
    .find((item) => item.season === season)
    ?.races.find((item) => item.round === round);
  const requested = cutoffSchema.safeParse(search.get("cutoff"));
  const cutoff: Cutoff | null = race
    ? requested.success &&
      race.cutoffs.some((entry) => entry.cutoff === requested.data && entry.available)
      ? requested.data
      : race.latest_cutoff
    : null;
  const entry = race?.cutoffs.find((item) => item.cutoff === cutoff);
  const path = entry?.available ? (entry.path ?? null) : null;
  const snapshot = useResource(path, () => loadSnapshot(path ?? ""));
  const tabParam = search.get("tab");
  const tab: TabId = isTab(tabParam) ? tabParam : "order";

  const update = (changes: Record<string, string>) => {
    const next = new URLSearchParams(search);
    for (const [name, value] of Object.entries(changes)) next.set(name, value);
    setSearch(next, { replace: true });
  };

  if (!race || !Number.isInteger(season) || !Number.isInteger(round)) {
    return (
      <main id="main" className="page">
        <EmptyState title="No prediction exported for this race">
          <p>
            Only races with a published development prediction appear here. See all exported
            predictions on the predictions page.
          </p>
        </EmptyState>
      </main>
    );
  }

  return (
    <>
      <RaceNavigator index={index} season={season} round={round} />
      <main id="main" className="page race-page">
        {snapshot.status === "loading" ? <LoadingState /> : null}
        {snapshot.status === "error" ? <ErrorState error={snapshot.error} /> : null}
        {snapshot.status === "ready" ? (
          <>
            <EventHero snapshot={snapshot.data} />
            <CutoffNav
              cutoffs={race.cutoffs}
              active={snapshot.data.identity.cutoff}
              onSelect={(value) => update({ cutoff: value })}
            />
            <Tabs
              label="Prediction views"
              idPrefix="race"
              tabs={[...RACE_TABS]}
              active={tab}
              onChange={(value) => update({ tab: value })}
            >
              {tab === "order" ? <PredictedOrderTable snapshot={snapshot.data} /> : null}
              {tab === "probabilities" ? (
                <>
                  <RaceProbabilitiesTable snapshot={snapshot.data} />
                  <WeekendEvolution race={race} current={snapshot.data} />
                </>
              ) : null}
              {tab === "wdc" ? (
                <ChampionshipTable kind="wdc" championship={snapshot.data.championship} />
              ) : null}
              {tab === "wcc" ? (
                <ChampionshipTable kind="wcc" championship={snapshot.data.championship} />
              ) : null}
              {tab === "result" ? <ActualResultPanel snapshot={snapshot.data} /> : null}
              {tab === "model" ? <ModelDetails snapshot={snapshot.data} /> : null}
            </Tabs>
          </>
        ) : null}
      </main>
    </>
  );
}
