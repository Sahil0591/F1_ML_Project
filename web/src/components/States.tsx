import type { ReactNode } from "react";
import type { LoadError, LoadErrorKind } from "../data/api";

export function LoadingState({ label = "Loading predictions" }: { label?: string }) {
  return (
    <div className="skeleton" role="status" aria-live="polite" aria-busy="true">
      <span className="visually-hidden">{label}</span>
      <div className="skeleton-line skeleton-title" />
      <div className="skeleton-line skeleton-meta" />
      {Array.from({ length: 8 }, (_, index) => (
        <div key={index} className="skeleton-line skeleton-row" />
      ))}
    </div>
  );
}

const ERROR_COPY: Record<LoadErrorKind, { title: string; body: string }> = {
  missing: {
    title: "Prediction artifact not found",
    body: "This file has not been exported yet. Run the export command after a prediction run.",
  },
  malformed: {
    title: "Prediction file failed validation",
    body: "The exported JSON does not match the expected schema, so nothing from it is shown.",
  },
  unsupported: {
    title: "Unsupported artifact version",
    body: "This export was written for a different schema version. Re-run the exporter with the current project code.",
  },
  network: {
    title: "Could not load predictions",
    body: "The data request failed. Check that the site is served together with its data directory.",
  },
};

export function ErrorState({ error }: { error: LoadError }) {
  const copy = ERROR_COPY[error.kind];
  return (
    <div className="state state-error" role="alert" data-error-kind={error.kind}>
      <p className="state-kicker">Error</p>
      <h2>{copy.title}</h2>
      <p>{copy.body}</p>
      <p className="state-detail">{error.message}</p>
      <pre className="state-command">python -m f1_ml_predictor export-web</pre>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="state state-empty">
      <h3>{title}</h3>
      {children}
    </div>
  );
}
