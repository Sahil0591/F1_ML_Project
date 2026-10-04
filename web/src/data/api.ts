import type { z } from "zod";
import { SCHEMA_VERSION, indexSchema, seasonSchema, snapshotSchema } from "./schema";
import type { ExportIndex, Season, Snapshot } from "./schema";

export type LoadErrorKind = "missing" | "malformed" | "unsupported" | "network";

/** A data file could not be shown; `kind` selects the error state the page renders. */
export class LoadError extends Error {
  readonly kind: LoadErrorKind;
  readonly path: string;

  constructor(kind: LoadErrorKind, path: string, message: string) {
    super(message);
    this.name = "LoadError";
    this.kind = kind;
    this.path = path;
  }
}

export function dataUrl(path: string): string {
  return `${import.meta.env.BASE_URL}data/${path}`;
}

async function fetchJson(path: string): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(dataUrl(path), { headers: { Accept: "application/json" } });
  } catch (error) {
    throw new LoadError("network", path, `Could not reach ${path}: ${String(error)}`);
  }
  if (response.status === 404) {
    throw new LoadError("missing", path, `${path} has not been exported`);
  }
  if (!response.ok) {
    throw new LoadError("network", path, `${path} returned HTTP ${response.status}`);
  }
  const text = await response.text();
  try {
    return JSON.parse(text) as unknown;
  } catch {
    // Vite serves index.html for unknown paths in development, which is not JSON.
    throw new LoadError(
      text.trimStart().startsWith("<") ? "missing" : "malformed",
      path,
      `${path} is not valid JSON`,
    );
  }
}

/** Version first, so an older or newer export reads as unsupported rather than broken. */
export function parseDocument<T>(path: string, raw: unknown, schema: z.ZodType<T>): T {
  if (typeof raw !== "object" || raw === null || !("schema_version" in raw)) {
    throw new LoadError("malformed", path, `${path} has no schema_version`);
  }
  const version = (raw as { schema_version: unknown }).schema_version;
  if (version !== SCHEMA_VERSION) {
    throw new LoadError(
      "unsupported",
      path,
      `${path} uses export schema ${String(version)}; this app reads schema ${SCHEMA_VERSION}`,
    );
  }
  const result = schema.safeParse(raw);
  if (!result.success) {
    const issue = result.error.issues[0];
    const where = issue ? issue.path.join(".") : "";
    throw new LoadError(
      "malformed",
      path,
      `${path} does not match the export schema${where ? ` at ${where}` : ""}`,
    );
  }
  return result.data;
}

const cache = new Map<string, Promise<unknown>>();

function load<T>(path: string, schema: z.ZodType<T>): Promise<T> {
  const cached = cache.get(path);
  if (cached) return cached as Promise<T>;
  const promise = fetchJson(path).then((raw) => parseDocument(path, raw, schema));
  promise.catch(() => cache.delete(path));
  cache.set(path, promise);
  return promise;
}

export const loadIndex = (): Promise<ExportIndex> => load("index.json", indexSchema);
export const loadSnapshot = (path: string): Promise<Snapshot> => load(path, snapshotSchema);
export const loadSeason = (path: string): Promise<Season> => load(path, seasonSchema);

export function clearCache(): void {
  cache.clear();
}
