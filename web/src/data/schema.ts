/**
 * Runtime schemas for the JSON written by `python -m f1_ml_predictor export-web`.
 * The Python exporter is the source of truth; these mirror its output so a
 * malformed or unexpected file is rejected instead of rendered.
 */
import { z } from "zod";

export const SCHEMA_VERSION = 1;

export const CUTOFFS = ["pre_weekend", "post_practice", "post_qualifying", "pre_race"] as const;
export const cutoffSchema = z.enum(CUTOFFS);
export type Cutoff = z.infer<typeof cutoffSchema>;

const probability = z.number().min(0).max(1);
const timestamp = z.string().min(1);

export const cutoffEntrySchema = z.object({
  cutoff: cutoffSchema,
  available: z.boolean(),
  path: z.string().nullable(),
  run_id: z.string().nullable(),
  prediction_timestamp: timestamp.nullable(),
  predicted_order_available: z.boolean(),
  superseded_run_ids: z.array(z.string()),
});

export const raceEntrySchema = z.object({
  season: z.number().int(),
  round: z.number().int(),
  event_id: z.string(),
  race_name: z.string(),
  circuit_name: z.string(),
  locality: z.string().nullable(),
  country: z.string().nullable(),
  race_start: timestamp,
  actual_result_available: z.boolean(),
  latest_cutoff: cutoffSchema,
  cutoffs: z.array(cutoffEntrySchema),
});

export const indexSchema = z.object({
  schema_version: z.literal(SCHEMA_VERSION),
  kind: z.literal("index"),
  cutoffs: z.array(cutoffSchema),
  latest: z.object({
    season: z.number().int(),
    round: z.number().int(),
    cutoff: cutoffSchema,
    path: z.string(),
  }),
  seasons: z.array(
    z.object({
      season: z.number().int(),
      total_rounds: z.number().int(),
      season_path: z.string(),
      final_results_available: z.boolean(),
      races: z.array(raceEntrySchema),
    }),
  ),
  excluded_runs: z.array(z.object({ path: z.string(), reason: z.string() })),
});

export const raceDriverSchema = z.object({
  driver_id: z.string(),
  driver_name: z.string(),
  driver_code: z.string(),
  constructor_id: z.string(),
  constructor_name: z.string(),
  predicted_position: z.number().int().positive().nullable(),
  winner_probability: probability,
  podium_probability: probability,
  dnf_probability: probability,
  dnf_model_probability: probability,
  expected_position: z.number(),
  most_likely_position: z.number().int().positive(),
  clean_expected_position: z.number().nullable(),
  clean_most_likely_position: z.number().int().positive().nullable(),
  position_interval_80: z.tuple([z.number().int(), z.number().int()]),
  winner_draws: z.number().int().nonnegative(),
});

export const standingEntrySchema = z.object({
  id: z.string(),
  name: z.string(),
  code: z.string().nullable(),
  constructor_id: z.string().nullable(),
  constructor_name: z.string().nullable(),
  points_now: z.number(),
  expected_final_points: z.number(),
  title_probability: probability,
  tied_for_title_probability: probability.nullable(),
  top3_probability: z.number().min(0).max(1 + 1e-9),
  most_likely_position: z.number().int().positive(),
  final_position_probability: z.array(probability),
  projected_position: z.number().int().positive(),
  fixed_strength_title_probability: probability.nullable(),
});

const titleTableSchema = z.object({
  unresolved_tie_probability: probability,
  entries: z.array(standingEntrySchema),
});

export const championshipSchema = z.object({
  season: z.number().int(),
  status: z.string(),
  simulations: z.number().int(),
  worlds: z.number().int(),
  orders_per_world: z.number().int(),
  monte_carlo_standard_error_max: z.number(),
  standings_available_at: timestamp,
  standings_notes: z.array(z.string()),
  remaining_sessions: z.array(
    z.object({
      event_id: z.string(),
      race_name: z.string(),
      session: z.string(),
      scheduled_at: timestamp,
    }),
  ),
  assumptions: z.array(z.string()),
  limitations: z.array(z.string()),
  candidate_sensitivity: z.record(
    z.string(),
    z.object({ wdc: z.record(z.string(), probability), wcc: z.record(z.string(), probability) }),
  ),
  wdc: titleTableSchema,
  wcc: titleTableSchema,
});

const metricsSchema = z.object({
  winner_log_loss: z.number(),
  winner_brier: z.number(),
  top1_accuracy: z.number(),
  podium_brier: z.number(),
  finish_mae: z.number(),
});

export const classificationRowSchema = z.object({
  position: z.number().int().nullable(),
  position_text: z.string().nullable(),
  classified: z.boolean(),
  driver_id: z.string(),
  driver_name: z.string(),
  driver_code: z.string(),
  constructor_id: z.string().nullable(),
  constructor_name: z.string(),
  grid: z.number().int().nullable(),
  laps: z.number().int().nullable(),
  points: z.number().nullable(),
  status: z.string().nullable(),
  predicted_position: z.number().int().nullable(),
  position_delta: z.number().int().nullable(),
  clean_expected_position: z.number().nullable(),
  clean_expected_error: z.number().nullable(),
  winner_probability: probability.nullable(),
  podium_probability: probability.nullable(),
  dnf_model_probability: probability.nullable(),
});

export const actualResultSchema = z.object({
  source_path: z.string(),
  source_sha256: z.string().nullable(),
  classification: z.array(classificationRowSchema),
  summary: z.object({
    winner_id: z.string().nullable(),
    winner_probability: probability.nullable(),
    predicted_winner_id: z.string().nullable(),
    podium: z.array(z.object({ driver_id: z.string(), podium_probability: probability.nullable() })),
    retirements: z.array(
      z.object({
        driver_id: z.string(),
        status: z.string().nullable(),
        dnf_model_probability: probability.nullable(),
      }),
    ),
    predicted_position_mae: z.number().nullable(),
    clean_expected_mae: z.number().nullable(),
    exact_position_hits: z.number().int(),
    classified_finishers: z.number().int(),
  }),
});

export const snapshotSchema = z.object({
  schema_version: z.literal(SCHEMA_VERSION),
  kind: z.literal("race_snapshot"),
  identity: z.object({
    season: z.number().int(),
    round: z.number().int(),
    cutoff: cutoffSchema,
    run_id: z.string(),
    methodology: z.string(),
    validation_status: z.string(),
    development_only: z.boolean(),
    validated_forecast: z.boolean(),
    prediction_timestamp: timestamp,
    created_at: timestamp,
    generated_after_race_start: z.boolean(),
    git_commit: z.string().nullable(),
    source_directory: z.string(),
  }),
  warning: z.string(),
  event: z.object({
    event_id: z.string(),
    race_name: z.string(),
    circuit_id: z.string(),
    circuit_name: z.string(),
    locality: z.string().nullable(),
    country: z.string().nullable(),
    race_start: timestamp,
    first_practice: timestamp.nullable(),
    qualifying_start: timestamp.nullable(),
    sprint_weekend: z.boolean(),
  }),
  status: z.object({
    ood_status: z.string(),
    ood_reasons: z.array(z.string()),
    unseen_circuit: z.boolean(),
    training_circuits: z.number().int(),
    development_quality: z.string(),
    checks_passed: z.boolean(),
    notes: z.array(z.string()),
  }),
  race: z.object({
    draws: z.number().int().positive(),
    seed: z.number().int(),
    monte_carlo_resolution: z.number(),
    predicted_order_available: z.boolean(),
    dnf_field_wide: z.boolean(),
    drivers: z.array(raceDriverSchema).min(1),
  }),
  championship: championshipSchema,
  model: z.object({
    primary_model: z.string(),
    primary_members: z.array(z.string()),
    experimental_model: z.string(),
    calibration: z.record(
      z.string(),
      z.object({ temperature: z.number(), shrink: z.number(), source: z.string() }),
    ),
    dnf_model: z.string(),
    dnf_candidates: z.record(
      z.string(),
      z.object({
        brier_score: z.number(),
        log_loss: z.number(),
        ece: z.number(),
        labels: z.number().int(),
        mean_within_race_standard_deviation: z.number(),
      }),
    ),
    dnf_known_labels: z.number().int(),
    feature_contract: z.string(),
    protocol_version: z.string(),
    protocol_sha256_short: z.string(),
    dataset_version: z.string(),
    dataset_hash_short: z.string(),
    dnf_dataset_version: z.string(),
    masked_for_training: z.array(z.string()),
    devices: z.array(z.string()),
    sharpness: z.record(z.string(), z.number()),
    historical: z
      .object({
        outer_races: z.number().int(),
        primary: metricsSchema,
        baselines: z.record(z.string(), metricsSchema),
        formal_gate: z.record(z.string(), z.string()),
      })
      .nullable(),
  }),
  actual_result: actualResultSchema.nullable(),
});

const finalRowSchema = z.object({
  id: z.string(),
  name: z.string(),
  position: z.number().int(),
  points: z.number(),
  tied_on_points: z.boolean(),
});

const seasonComparisonRowSchema = z.object({
  id: z.string(),
  name: z.string(),
  final_position: z.number().int(),
  actual_points: z.number(),
  projected_position: z.number().int().nullable(),
  projected_final_points: z.number().nullable(),
  title_probability: probability.nullable(),
  points_error: z.number().nullable(),
  position_error: z.number().int().nullable(),
});

export const seasonSchema = z.object({
  schema_version: z.literal(SCHEMA_VERSION),
  kind: z.literal("season"),
  season: z.number().int(),
  total_rounds: z.number().int(),
  completed: z.boolean(),
  latest: z.object({
    round: z.number().int(),
    cutoff: cutoffSchema,
    path: z.string(),
    prediction_timestamp: timestamp,
  }),
  snapshots: z.array(
    z.object({
      round: z.number().int(),
      cutoff: cutoffSchema,
      path: z.string(),
      run_id: z.string(),
      prediction_timestamp: timestamp,
      wdc_title: z.record(z.string(), probability),
      wcc_title: z.record(z.string(), probability),
    }),
  ),
  final_standings: z
    .object({
      source: z.string(),
      scoring_ledger_sha256: z.string(),
      notes: z.array(z.string()),
      drivers: z.array(finalRowSchema),
      constructors: z.array(finalRowSchema),
    })
    .nullable(),
  final_comparison: z
    .object({
      drivers: z.array(seasonComparisonRowSchema),
      constructors: z.array(seasonComparisonRowSchema),
    })
    .nullable(),
});

export type ExportIndex = z.infer<typeof indexSchema>;
export type RaceEntry = z.infer<typeof raceEntrySchema>;
export type CutoffEntry = z.infer<typeof cutoffEntrySchema>;
export type Snapshot = z.infer<typeof snapshotSchema>;
export type RaceDriver = z.infer<typeof raceDriverSchema>;
export type StandingEntry = z.infer<typeof standingEntrySchema>;
export type Championship = z.infer<typeof championshipSchema>;
export type ActualResult = z.infer<typeof actualResultSchema>;
export type ClassificationRow = z.infer<typeof classificationRowSchema>;
export type Season = z.infer<typeof seasonSchema>;
export type SeasonComparisonRow = z.infer<typeof seasonComparisonRowSchema>;
