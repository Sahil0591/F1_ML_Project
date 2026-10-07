/**
 * Synthetic export documents for component tests. They follow the exporter's
 * schema but carry made-up engineering values, never real predictions.
 */
import { vi } from "vitest";
import type { Cutoff, ExportIndex, Season, Snapshot } from "../data/schema";
import { CUTOFFS, indexSchema, seasonSchema, snapshotSchema } from "../data/schema";

const TEAMS = [
  "alpha",
  "bravo",
  "charlie",
  "delta",
  "echo",
  "foxtrot",
  "golf",
  "hotel",
  "india",
  "juliet",
  "kilo",
];

export const DRIVER_COUNT = 22;

function finishDistribution(rank: number, count: number): number[] {
  // Geometric-like weights centred on the driver's rank, normalised to sum to 1.
  const weights = Array.from({ length: count }, (_, place) => Math.pow(0.55, Math.abs(place - rank)));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map((value) => value / total);
}

export function makeSnapshot(
  options: {
    cutoff?: Cutoff;
    withOrder?: boolean;
    withResult?: boolean;
    ood?: boolean;
    winnerShift?: number;
  } = {},
): Snapshot {
  const { cutoff = "pre_weekend", withOrder = true, withResult = false, ood = true } = options;
  const shift = options.winnerShift ?? 0;
  const drivers = Array.from({ length: DRIVER_COUNT }, (_, index) => {
    const rank = (index + shift) % DRIVER_COUNT;
    const finish = finishDistribution(rank, DRIVER_COUNT);
    const team = TEAMS[Math.floor(index / 2)] ?? "kilo";
    return {
      driver_id: `driver_${String(index + 1).padStart(2, "0")}`,
      driver_name: `Test Driver${index + 1}`,
      driver_code: `D${String(index + 1).padStart(2, "0")}`,
      constructor_id: team,
      constructor_name: `Team ${team}`,
      predicted_position: withOrder ? rank + 1 : null,
      winner_probability: index === DRIVER_COUNT - 1 ? 0 : (finish[0] ?? 0),
      podium_probability: (finish[0] ?? 0) + (finish[1] ?? 0) + (finish[2] ?? 0),
      dnf_probability: 0.12,
      dnf_model_probability: 0.1253315649867374,
      expected_position: rank + 1.5,
      most_likely_position: rank + 1,
      clean_expected_position: withOrder ? rank + 1.25 : null,
      clean_most_likely_position: withOrder ? rank + 1 : null,
      position_interval_80: [Math.max(1, rank - 2), Math.min(DRIVER_COUNT, rank + 4)] as [
        number,
        number,
      ],
      winner_draws: 100,
    };
  });
  const driverIds = drivers.map((driver) => driver.driver_id);
  const places = (index: number, size: number) =>
    Array.from({ length: size }, (_, place) => (place === index ? 1 : 0));
  const snapshot = {
    schema_version: 1,
    kind: "race_snapshot",
    identity: {
      season: 2026,
      round: 16,
      cutoff,
      session: cutoff === "post_sprint_qualifying" ? "sprint" : "race",
      run_id: `run-${cutoff}`,
      methodology: "cutoff-specific-v3",
      validation_status: "development_only",
      development_only: true,
      validated_forecast: false,
      prediction_timestamp: "2026-10-02T09:35:40+00:00",
      created_at: "2026-10-02T09:39:41+00:00",
      generated_after_race_start: false,
      git_commit: "65c35d642585f8e566324a66a063db785f7cae63",
      source_directory: "data/predictions/development/next_race/fixture",
    },
    warning: "DEVELOPMENT ONLY. Fixture warning.",
    event: {
      event_id: "season=2026/round=16",
      race_name: "Fixture Grand Prix",
      circuit_id: "fixture_ring",
      circuit_name: "Fixture Ring",
      locality: "Testville",
      country: "Nowhere",
      race_start: "2026-10-04T07:00:00+00:00",
      first_practice: "2026-10-02T04:30:00+00:00",
      qualifying_start: "2026-10-03T08:00:00+00:00",
      sprint_weekend: false,
    },
    status: {
      ood_status: ood ? "out_of_distribution" : "in_distribution",
      ood_reasons: ood ? ["circuit fixture_ring is absent from the Gold training races"] : [],
      unseen_circuit: ood,
      training_circuits: 26,
      development_quality: "pass",
      checks_passed: true,
      notes: [],
    },
    race: {
      draws: 65536,
      seed: 42,
      monte_carlo_resolution: 1 / 65536,
      predicted_order_available: withOrder,
      dnf_field_wide: true,
      drivers,
    },
    championship: {
      season: 2026,
      status: "engineering_only",
      simulations: 100000,
      worlds: 1000,
      orders_per_world: 16,
      monte_carlo_standard_error_max: 0.0158,
      standings_available_at: "2026-09-26T17:20:00+00:00",
      standings_notes: [],
      remaining_sessions: [],
      assumptions: ["Fixture assumption."],
      limitations: [],
      candidate_sensitivity: {},
      wdc: {
        unresolved_tie_probability: 0.0004,
        entries: driverIds.map((id, index) => ({
          id,
          name: `Test Driver${index + 1}`,
          code: `D${String(index + 1).padStart(2, "0")}`,
          constructor_id: TEAMS[Math.floor(index / 2)] ?? "kilo",
          constructor_name: `Team ${TEAMS[Math.floor(index / 2)] ?? "kilo"}`,
          points_now: 300 - index * 10,
          expected_final_points: 420.5 - index * 12,
          title_probability: index === 0 ? 0.77123 : index === 1 ? 0.22877 : 0,
          tied_for_title_probability: 0,
          top3_probability: index < 3 ? 0.9 : 0.004,
          most_likely_position: index + 1,
          final_position_probability: places(index, DRIVER_COUNT),
          projected_position: index + 1,
          fixed_strength_title_probability: index === 0 ? 0.9 : 0,
        })),
      },
      wcc: {
        unresolved_tie_probability: 0,
        entries: TEAMS.map((id, index) => ({
          id,
          name: `Team ${id}`,
          code: null,
          constructor_id: null,
          constructor_name: null,
          points_now: 500 - index * 40,
          expected_final_points: 700.25 - index * 50,
          title_probability: index === 0 ? 0.99407 : index === 1 ? 0.00593 : 0,
          tied_for_title_probability: 0,
          top3_probability: index < 3 ? 1 : 0,
          most_likely_position: index + 1,
          final_position_probability: places(index, TEAMS.length),
          projected_position: index + 1,
          fixed_strength_title_probability: null,
        })),
      },
    },
    model: {
      primary_model: "ensemble_all",
      primary_members: ["logistic_pl", "hist"],
      experimental_model: "lightgbm",
      calibration: { logistic_pl: { temperature: 1, shrink: 0, source: "shared" } },
      dnf_model: "prior",
      dnf_candidates: {
        prior: {
          brier_score: 0.111,
          log_loss: 0.38,
          ece: 0.02,
          labels: 1506,
          mean_within_race_standard_deviation: 0,
        },
      },
      dnf_known_labels: 1506,
      feature_contract: "cutoff-contracts-v2",
      protocol_version: "gold-cutoff-specific-v3",
      protocol_sha256_short: "c3fc3c0cd48b",
      dataset_version: "dataset-ee66db76",
      dataset_hash_short: "ee66db760961",
      dnf_dataset_version: "dataset-d03ac008",
      masked_for_training: [],
      devices: ["cpu"],
      sharpness: { maximum_win_probability: 0.23 },
      historical: {
        outer_races: 92,
        primary: {
          winner_log_loss: 1.5645,
          winner_brier: 0.6995,
          top1_accuracy: 0.39,
          podium_brier: 0.0863,
          finish_mae: 2.987,
        },
        baselines: {},
        formal_gate: { winner: "no_selection" },
      },
    },
    actual_result: withResult ? makeResult(drivers) : null,
  };
  return snapshotSchema.parse(snapshot);
}

function makeResult(drivers: Snapshot["race"]["drivers"]): Snapshot["actual_result"] {
  // Finishing order: predicted P2 wins, predicted P1 second, last driver retires.
  const ordered = [...drivers].sort(
    (a, b) => (a.predicted_position ?? 0) - (b.predicted_position ?? 0),
  );
  const [first, second, ...rest] = ordered;
  if (!first || !second) throw new Error("fixture needs drivers");
  const finishing = [second, first, ...rest];
  const classification = finishing.map((driver, index) => {
    const retired = index === finishing.length - 1;
    const position = index + 1;
    return {
      position,
      position_text: retired ? "R" : String(position),
      classified: !retired,
      driver_id: driver.driver_id,
      driver_name: driver.driver_name,
      driver_code: driver.driver_code,
      constructor_id: driver.constructor_id,
      constructor_name: driver.constructor_name,
      grid: index + 1,
      laps: retired ? 20 : 56,
      points: index < 10 ? 25 - index * 2 : 0,
      status: retired ? "Retired" : "Finished",
      predicted_position: driver.predicted_position,
      position_delta: retired || driver.predicted_position === null ? null : driver.predicted_position - position,
      clean_expected_position: driver.clean_expected_position,
      clean_expected_error:
        retired || driver.clean_expected_position === null ? null : position - driver.clean_expected_position,
      winner_probability: driver.winner_probability,
      podium_probability: driver.podium_probability,
      dnf_model_probability: driver.dnf_model_probability,
    };
  });
  const last = classification[classification.length - 1];
  return {
    source_path: "data/normalized/season=2026/round=16/results.parquet",
    source_sha256: "abcdef0123456789",
    classification,
    summary: {
      winner_id: second.driver_id,
      winner_probability: second.winner_probability,
      predicted_winner_id: first.driver_id,
      podium: classification.slice(0, 3).map((row) => ({
        driver_id: row.driver_id,
        podium_probability: row.podium_probability,
      })),
      retirements: last
        ? [
            {
              driver_id: last.driver_id,
              status: last.status,
              dnf_model_probability: last.dnf_model_probability,
            },
          ]
        : [],
      predicted_position_mae: 0.1,
      clean_expected_mae: 0.3,
      exact_position_hits: 19,
      classified_finishers: 21,
    },
  };
}

export function makeIndex(available: Cutoff[] = ["pre_weekend", "post_qualifying"]): ExportIndex {
  return indexSchema.parse({
    schema_version: 1,
    kind: "index",
    cutoffs: CUTOFFS,
    latest: {
      season: 2026,
      round: 16,
      cutoff: available[available.length - 1],
      path: `2026/round-16/${available[available.length - 1]}.json`,
    },
    seasons: [
      {
        season: 2026,
        total_rounds: 22,
        season_path: "2026/season.json",
        final_results_available: false,
        races: [
          {
            season: 2026,
            round: 16,
            event_id: "season=2026/round=16",
            race_name: "Fixture Grand Prix",
            circuit_name: "Fixture Ring",
            locality: "Testville",
            country: "Nowhere",
            race_start: "2026-10-04T07:00:00+00:00",
            actual_result_available: false,
            latest_cutoff: available[available.length - 1],
            cutoffs: CUTOFFS.map(
              (cutoff) => ({
                cutoff,
                session: cutoff === "post_sprint_qualifying" ? "sprint" : "race",
                available: available.includes(cutoff),
                path: available.includes(cutoff) ? `2026/round-16/${cutoff}.json` : null,
                run_id: available.includes(cutoff) ? `run-${cutoff}` : null,
                prediction_timestamp: available.includes(cutoff) ? "2026-10-02T09:35:40+00:00" : null,
                predicted_order_available: available.includes(cutoff),
                superseded_run_ids: [],
              }),
            ),
          },
        ],
      },
    ],
    excluded_runs: [],
  });
}

export function makeSeason(completed = false): Season {
  const snapshot = makeSnapshot();
  return seasonSchema.parse({
    schema_version: 1,
    kind: "season",
    season: 2026,
    total_rounds: 22,
    completed,
    latest: {
      round: 16,
      cutoff: "pre_weekend",
      path: "2026/round-16/pre_weekend.json",
      prediction_timestamp: "2026-10-02T09:35:40+00:00",
    },
    snapshots: [
      {
        round: 16,
        cutoff: "pre_weekend",
        path: "2026/round-16/pre_weekend.json",
        run_id: "run-pre_weekend",
        prediction_timestamp: "2026-10-02T09:35:40+00:00",
        wdc_title: Object.fromEntries(
          snapshot.championship.wdc.entries.map((entry) => [entry.id, entry.title_probability]),
        ),
        wcc_title: Object.fromEntries(
          snapshot.championship.wcc.entries.map((entry) => [entry.id, entry.title_probability]),
        ),
      },
    ],
    final_standings: completed
      ? {
          source: "audited scoring ledger",
          scoring_ledger_sha256: "f".repeat(64),
          notes: [],
          drivers: [
            { id: "driver_02", name: "Test Driver2", position: 1, points: 450, tied_on_points: false },
            { id: "driver_01", name: "Test Driver1", position: 2, points: 440, tied_on_points: false },
          ],
          constructors: [
            { id: "alpha", name: "Team alpha", position: 1, points: 800, tied_on_points: false },
          ],
        }
      : null,
    final_comparison: completed
      ? {
          drivers: [
            {
              id: "driver_02",
              name: "Test Driver2",
              final_position: 1,
              actual_points: 450,
              projected_position: 2,
              projected_final_points: 408.5,
              title_probability: 0.22877,
              points_error: 41.5,
              position_error: 1,
            },
            {
              id: "driver_01",
              name: "Test Driver1",
              final_position: 2,
              actual_points: 440,
              projected_position: 1,
              projected_final_points: 420.5,
              title_probability: 0.77123,
              points_error: 19.5,
              position_error: -1,
            },
          ],
          constructors: [
            {
              id: "alpha",
              name: "Team alpha",
              final_position: 1,
              actual_points: 800,
              projected_position: 1,
              projected_final_points: 700.25,
              title_probability: 0.99407,
              points_error: 99.75,
              position_error: 0,
            },
          ],
        }
      : null,
  });
}

/** Serve documents by data path; anything else is a 404 like a static host. */
export function stubFetch(documents: Record<string, unknown | string>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const path = url.slice(url.indexOf("/data/") + "/data/".length);
    if (!(path in documents)) return new Response("not found", { status: 404 });
    const body = documents[path];
    return new Response(typeof body === "string" ? body : JSON.stringify(body), { status: 200 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
