import type { Snapshot } from "../data/schema";
import { cutoffLabel, formatProbability, humanize } from "../format";

function Metric({ label, value, digits = 3 }: { label: string; value: number | undefined; digits?: number }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value === undefined ? "n/a" : value.toFixed(digits)}</dd>
    </div>
  );
}

export function ModelDetails({ snapshot }: { snapshot: Snapshot }) {
  const { model, identity, status, race, championship } = snapshot;
  const historical = model.historical;
  const logistic = historical?.baselines.logistic;
  const dnfValue = race.drivers[0]?.dnf_model_probability;
  return (
    <section aria-labelledby="model-title">
      <div className="section-head">
        <h2 id="model-title">Model details</h2>
        <p className="section-lede">
          Provenance and evidence for this snapshot. {snapshot.warning}
        </p>
      </div>
      <details className="details" open>
        <summary>Run and data</summary>
        <dl className="kv">
          <div>
            <dt>Status</dt>
            <dd>
              {identity.validation_status}
              {identity.validated_forecast ? "" : ", not a validated forecast"}
            </dd>
          </div>
          <div>
            <dt>Cutoff</dt>
            <dd>{cutoffLabel(identity.cutoff)}</dd>
          </div>
          <div>
            <dt>Run ID</dt>
            <dd>
              <code>{identity.run_id}</code>
            </dd>
          </div>
          <div>
            <dt>Methodology</dt>
            <dd>{identity.methodology}</dd>
          </div>
          <div>
            <dt>Protocol</dt>
            <dd>
              {model.protocol_version} (<code>{model.protocol_sha256_short}</code>)
            </dd>
          </div>
          <div>
            <dt>Feature contract</dt>
            <dd>{model.feature_contract}</dd>
          </div>
          <div>
            <dt>Gold dataset</dt>
            <dd>
              <code title={model.dataset_version}>{model.dataset_hash_short}</code>
            </dd>
          </div>
          <div>
            <dt>Code commit</dt>
            <dd>
              <code>{identity.git_commit ? identity.git_commit.slice(0, 7) : "n/a"}</code>
            </dd>
          </div>
        </dl>
      </details>
      <details className="details">
        <summary>Models and calibration</summary>
        <dl className="kv">
          <div>
            <dt>Primary model</dt>
            <dd>{humanize(model.primary_model)}</dd>
          </div>
          <div>
            <dt>Ensemble members</dt>
            <dd>{model.primary_members.map(humanize).join(", ")}</dd>
          </div>
          <div>
            <dt>Experimental boosting model</dt>
            <dd>{model.experimental_model}</dd>
          </div>
          <div>
            <dt>Execution</dt>
            <dd>{model.devices.join(", ")}</dd>
          </div>
        </dl>
        <div className="table-scroll" tabIndex={0} role="region" aria-label="Calibration by member">
          <table className="results-table compact">
            <thead>
              <tr>
                <th scope="col">Member</th>
                <th scope="col" className="num">
                  Temperature
                </th>
                <th scope="col" className="num">
                  Prior shrinkage
                </th>
                <th scope="col">Source</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(model.calibration).map(([name, values]) => (
                <tr key={name}>
                  <th scope="row">{humanize(name)}</th>
                  <td className="num">{values.temperature}</td>
                  <td className="num">{values.shrink}</td>
                  <td>{humanize(values.source)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <details className="details">
        <summary>Historical evidence at this cutoff</summary>
        {historical ? (
          <>
            <p className="note">
              Out-of-fold results on {historical.outer_races} chronological outer Gold races. Lower
              is better except top-1 accuracy. Formal gate:{" "}
              {Object.entries(historical.formal_gate)
                .map(([task, value]) => `${humanize(task)} ${humanize(value)}`)
                .join(", ")}
              .
            </p>
            <dl className="kv">
              <Metric label="Winner log loss" value={historical.primary.winner_log_loss} />
              <Metric label="Logistic baseline log loss" value={logistic?.winner_log_loss} />
              <Metric label="Podium Brier" value={historical.primary.podium_brier} digits={4} />
              <Metric label="Logistic podium Brier" value={logistic?.podium_brier} digits={4} />
              <Metric label="Finish MAE" value={historical.primary.finish_mae} digits={2} />
              <Metric label="Logistic finish MAE" value={logistic?.finish_mae} digits={2} />
            </dl>
          </>
        ) : (
          <p className="note">The cutoff evaluation file is not available locally.</p>
        )}
      </details>
      <details className="details">
        <summary>Simulation, DNF and distribution checks</summary>
        <dl className="kv">
          <div>
            <dt>Race draws</dt>
            <dd>{race.draws.toLocaleString("en-GB")}</dd>
          </div>
          <div>
            <dt>Season simulations</dt>
            <dd>{championship.simulations.toLocaleString("en-GB")}</dd>
          </div>
          <div>
            <dt>Uncertainty worlds</dt>
            <dd>
              {championship.worlds.toLocaleString("en-GB")} ({championship.orders_per_world} orders
              each)
            </dd>
          </div>
          <div>
            <dt>Title Monte Carlo SE max</dt>
            <dd>{championship.monte_carlo_standard_error_max.toFixed(3)}</dd>
          </div>
          <div>
            <dt>DNF model</dt>
            <dd>
              {model.dnf_model}
              {race.dnf_field_wide && dnfValue !== undefined
                ? `, field-wide rate ${formatProbability(dnfValue)}`
                : ", individual rates"}{" "}
              ({model.dnf_known_labels.toLocaleString("en-GB")} audited labels)
            </dd>
          </div>
          <div>
            <dt>Out of distribution</dt>
            <dd>{humanize(status.ood_status)}</dd>
          </div>
          <div>
            <dt>Unseen circuit</dt>
            <dd>
              {status.unseen_circuit ? "Yes" : "No"} ({status.training_circuits} training circuits)
            </dd>
          </div>
          <div>
            <dt>Development quality</dt>
            <dd>{status.development_quality}</dd>
          </div>
          <div>
            <dt>Coherence and leakage checks</dt>
            <dd>{status.checks_passed ? "All passed" : "Failed"}</dd>
          </div>
        </dl>
      </details>
      <details className="details">
        <summary>Limitations</summary>
        <ul className="plain-list">
          {status.ood_reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
          {status.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
          {championship.standings_notes.map((note) => (
            <li key={note}>Starting points: {note}</li>
          ))}
          {championship.assumptions.map((item) => (
            <li key={item}>{item}</li>
          ))}
          {model.masked_for_training.length ? (
            <li>
              Hidden from training because no driver has them at this cutoff:{" "}
              {model.masked_for_training.join(", ")}
            </li>
          ) : null}
        </ul>
      </details>
    </section>
  );
}
