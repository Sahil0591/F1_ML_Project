import type { Snapshot } from "../data/schema";
import { cutoffLabel, formatDate, formatDateTime } from "../format";

export function Badges({ snapshot }: { snapshot: Snapshot }) {
  const { identity, status } = snapshot;
  return (
    <ul className="badges" aria-label="Prediction status">
      {identity.development_only ? <li className="badge badge-dev">Development only</li> : null}
      <li className="badge badge-cutoff">{cutoffLabel(identity.cutoff)}</li>
      {status.ood_status === "out_of_distribution" ? (
        <li className="badge badge-warn">Out of distribution</li>
      ) : null}
      {status.unseen_circuit ? <li className="badge badge-warn">Unseen circuit</li> : null}
      {identity.generated_after_race_start ? (
        <li className="badge badge-outline">Generated after race start</li>
      ) : null}
      {!status.checks_passed ? <li className="badge badge-warn">Checks failed</li> : null}
    </ul>
  );
}

export function EventHero({ snapshot }: { snapshot: Snapshot }) {
  const { event, identity } = snapshot;
  const place = [event.locality, event.country].filter(Boolean).join(", ");
  return (
    <section className="hero" aria-labelledby="event-title">
      <div className="hero-inner">
        <p className="hero-kicker">
          {identity.season} <span aria-hidden="true">/</span> Round {identity.round}
          {event.sprint_weekend ? " / Sprint weekend" : ""}
        </p>
        <h1 id="event-title">{event.race_name}</h1>
        <dl className="hero-meta">
          <div>
            <dt>Circuit</dt>
            <dd>{event.circuit_name}</dd>
          </div>
          {place ? (
            <div>
              <dt>Location</dt>
              <dd>{place}</dd>
            </div>
          ) : null}
          <div>
            <dt>Race date</dt>
            <dd>
              <time dateTime={event.race_start} title={formatDateTime(event.race_start)}>
                {formatDate(event.race_start)}
              </time>
            </dd>
          </div>
          <div>
            <dt>Prediction cutoff</dt>
            <dd>
              {cutoffLabel(identity.cutoff)},{" "}
              <time dateTime={identity.prediction_timestamp}>
                {formatDateTime(identity.prediction_timestamp)}
              </time>
            </dd>
          </div>
          <div>
            <dt>Generated</dt>
            <dd>
              <time dateTime={identity.created_at}>{formatDateTime(identity.created_at)}</time>
            </dd>
          </div>
        </dl>
        <Badges snapshot={snapshot} />
        <p className="hero-warning">{snapshot.warning}</p>
        {identity.generated_after_race_start ? (
          <p className="hero-warning">
            This run was generated after the race started, as a replay at the original cutoff. The
            pipeline only admits data published before the cutoff
            {snapshot.status.checks_passed ? ", and its leakage checks passed." : "."}
          </p>
        ) : null}
      </div>
    </section>
  );
}
