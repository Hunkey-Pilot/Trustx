import type { CounterfactualResponse } from "@/lib/api";

interface CounterfactualAnalysisProps {
  value: CounterfactualResponse | null;
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US", {
    maximumSignificantDigits: 8,
  }).format(value);
}

function formatChange(value: number | null): string {
  if (value === null) return "Invalid scenario";
  return `${value > 0 ? "+" : ""}${formatNumber(value)}`;
}

export default function CounterfactualAnalysis({
  value,
}: CounterfactualAnalysisProps) {
  return (
    <div className="counterfactual-analysis">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Hypothetical model scenarios</p>
          <h2>Counterfactual Analysis</h2>
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
              <span>Original fraud probability</span>
              <strong>{formatNumber(value.original_fraud_probability)}</strong>
            </div>
          </div>

          {value.best_counterfactual ? (
            <p className="counterfactual-best-note">
              {value.best_counterfactual.description} Reduction: {formatNumber(value.best_counterfactual.probability_reduction)}.
            </p>
          ) : (
            <p className="counterfactual-no-result">No qualifying counterfactual found.</p>
          )}

          <div className="counterfactual-table-scroll">
            <table className="counterfactual-table">
              <thead>
                <tr>
                  <th>Scenario</th>
                  <th>Amount</th>
                  <th>Fraud Probability</th>
                  <th>Change</th>
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
                          ? formatNumber(scenario.fraud_probability)
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
        </>
      ) : (
        <p className="muted-copy">Not available</p>
      )}

      <p className="counterfactual-disclaimer">
        Counterfactual results are hypothetical model scenarios and do not guarantee that changing the transaction amount would prevent fraud.
      </p>
    </div>
  );
}