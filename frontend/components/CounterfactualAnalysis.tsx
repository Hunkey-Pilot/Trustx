import type { CounterfactualResponse } from "@/lib/api";
import { formatNumber } from "@/lib/format";

interface CounterfactualAnalysisProps {
  value: CounterfactualResponse | null;
}

function formatChange(value: number | null): string {
  if (value === null) return "Invalid scenario";
  return `${value > 0 ? "+" : ""}${formatNumber(value, 6)}`;
}

export default function CounterfactualAnalysis({
  value,
}: CounterfactualAnalysisProps) {
  return (
    <div className="counterfactual-analysis">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Hypothetical model scenarios</p>
          <h2>Counterfactual analysis</h2>
        </div>
        <span className={value ? "availability available" : "availability"}>
          {value ? "Available" : "Not available"}
        </span>
      </div>

      {value ? (
        <>
          <div className="counterfactual-originals">
            <div>
              <span>Original amount</span>
              <strong>{formatNumber(value.original_amount)}</strong>
            </div>
            <div>
              <span>Stored model score</span>
              <strong>{formatNumber(value.original_fraud_probability, 6)}</strong>
            </div>
            <div>
              <span>
                {value.baseline_source === "recomputed"
                  ? "Recomputed baseline (100%)"
                  : "Baseline (stored score)"}
              </span>
              <strong>
                {formatNumber(value.baseline_fraud_probability ?? value.original_fraud_probability, 6)}
              </strong>
            </div>
          </div>
          {value.baseline_matches_stored === false && (
            <p className="source-note">
              The recomputed baseline differs from the stored score because scenarios rebuild the
              balances consistently (balance-error features become zero). Changes below are
              measured against the recomputed baseline.
            </p>
          )}

          <div className="interpretation-block">
            <h3>Model output</h3>
            <p>
              {value.interpretation?.model_output.replace(/^MODEL OUTPUT:\s*/, "") ??
                (value.best_counterfactual
                  ? `${value.best_counterfactual.description} Reduction: ${formatNumber(value.best_counterfactual.probability_reduction, 6)}.`
                  : "No qualifying counterfactual found.")}
            </p>
          </div>

          <div className="counterfactual-table-scroll">
            <table className="counterfactual-table">
              <thead>
                <tr>
                  <th>Scenario</th>
                  <th>Amount</th>
                  <th>Model score</th>
                  <th>Change vs baseline</th>
                </tr>
              </thead>
              <tbody>
                {value.counterfactuals.map((scenario) => {
                  const isBest =
                    scenario.valid &&
                    value.best_counterfactual?.amount_ratio === scenario.amount_ratio;
                  return (
                    <tr
                      className={isBest ? "best-counterfactual-row" : undefined}
                      key={scenario.amount_ratio}
                    >
                      <th scope="row">
                        {Math.round(scenario.amount_ratio * 100)}%
                        {isBest && <span className="best-scenario-label">Best</span>}
                      </th>
                      <td>{formatNumber(scenario.amount)}</td>
                      <td>
                        {scenario.valid && scenario.fraud_probability !== null
                          ? formatNumber(scenario.fraud_probability, 6)
                          : "Invalid scenario"}
                      </td>
                      <td>
                        {scenario.valid
                          ? formatChange(scenario.probability_change)
                          : "Invalid scenario"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="interpretation-block recommendation-block">
            <h3>Investigation recommendation</h3>
            {value.interpretation?.investigation_recommendations.length ? (
              <ul className="recommendation-list">
                {value.interpretation.investigation_recommendations.map((action) => (
                  <li key={action.code}>
                    <strong>{action.label}</strong>
                    <span>{action.rationale}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted-copy">No specific investigation action suggested.</p>
            )}
          </div>
        </>
      ) : (
        <p className="muted-copy">Not available</p>
      )}

      <p className="counterfactual-disclaimer">
        {value?.interpretation?.disclaimer ??
          "Counterfactual scenarios are hypothetical model responses, not causal explanations. A lower amount does not make a transaction legitimate."}
      </p>
    </div>
  );
}
