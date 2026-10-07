import Link from "next/link";
import type {
  CounterfactualResponse,
  TransactionAnalysisResponse,
} from "@/lib/api";
import CounterfactualAnalysis from "@/components/CounterfactualAnalysis";
import BehavioralSignalsPanel from "@/components/BehavioralSignalsPanel";
import InvestigatorSummaryPanel from "@/components/InvestigatorSummaryPanel";
import NetworkAnalysisPanel from "@/components/NetworkAnalysisPanel";
import { MODEL_SCORE_DISCLAIMER, formatNumber } from "@/lib/format";

interface AnalysisResultProps {
  result: TransactionAnalysisResponse | null;
  counterfactual: CounterfactualResponse | null;
}

export default function AnalysisResult({
  result,
  counterfactual,
}: AnalysisResultProps) {
  if (!result) {
    return (
      <div className="empty-state" aria-live="polite">
        <span className="empty-mark" aria-hidden="true">
          TX
        </span>
        <p>Enter transaction details and click Analyze Transaction.</p>
      </div>
    );
  }

  return (
    <div className="results-content" aria-live="polite">
      {result.is_duplicate && (
        <p className="duplicate-notice" role="status">
          This transaction was already analyzed. The stored analysis is shown and no new record was
          created.
        </p>
      )}
      <section className="summary-section">
        <div className="result-title-row">
          <div>
            <p className="eyebrow">Analysis result</p>
            <h2>Risk assessment</h2>
          </div>
          <span className={`risk-tag risk-${result.risk_level.toLowerCase()}`}>
            {result.risk_level}
          </span>
        </div>

        <div className="metric-grid">
          <div className="metric metric-primary">
            <span>Model score</span>
            <strong>{formatNumber(result.fraud_probability)}</strong>
            <small>Uncalibrated model output</small>
          </div>
          <div className="metric">
            <span>Risk score</span>
            <strong>{formatNumber(result.risk_score)}</strong>
            <small>Model score x 100</small>
          </div>
          <div className="metric">
            <span>Anomaly signal</span>
            <strong>{formatNumber(result.anomaly_signal)}</strong>
            <small>Isolation Forest value (below 0 = more unusual); not part of the risk score</small>
          </div>
          <div className="metric">
            <span>Recommended action</span>
            <strong className="action-value">{result.recommended_action}</strong>
          </div>
        </div>
        <p className="score-disclaimer">{MODEL_SCORE_DISCLAIMER}</p>
        <Link
          className="view-link open-investigation-link"
          href={`/transactions/${encodeURIComponent(result.transaction_id)}`}
        >
          Open full investigation
        </Link>
      </section>

      <InvestigatorSummaryPanel value={result.investigator_summary} />

      <section className="detail-section shap-section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Model attribution</p>
            <h3>SHAP explanation</h3>
          </div>
          <span className="availability">
            {result.explanation.status === "available" ? "Available" : "Not available"}
          </span>
        </div>
        {result.explanation.status === "available" &&
        result.explanation.top_features.length > 0 ? (
          <div className="shap-list">
            {result.explanation.top_features.map((feature) => (
              <article className="shap-row" key={feature.feature}>
                <div className="shap-feature-name">{feature.feature}</div>
                <div className="shap-value">
                  <span>Value</span>
                  <strong>{formatNumber(feature.value)}</strong>
                </div>
                <div className="shap-value">
                  <span>SHAP</span>
                  <strong>{formatNumber(feature.shap_value)}</strong>
                </div>
                <div className="shap-direction">
                  {feature.direction.replaceAll("_", " ")}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <p className="muted-copy">Not available</p>
        )}
        <p className="source-note">SHAP describes how the model used its inputs; it is not causal proof.</p>
      </section>

      <section className="detail-section evidence-section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Supporting signals</p>
            <h3>Evidence</h3>
          </div>
          <span className="availability">{result.evidence.status}</span>
        </div>
        <div className="evidence-list">
          {result.evidence.items.map((item, index) => (
            <article className="evidence-row" key={`${item.category}-${item.type}-${index}`}>
              <div className="evidence-meta">
                <span className="evidence-category">{item.category.replaceAll("_", " ")}</span>
                <strong>{item.type.replaceAll("_", " ")}</strong>
              </div>
              <p>{item.description}</p>
              {item.value !== null && <span className="evidence-value">{formatNumber(item.value)}</span>}
            </article>
          ))}
        </div>
      </section>

      <BehavioralSignalsPanel value={result.behavioral_signals} />
      <NetworkAnalysisPanel value={result.network_signals} />

      <section className="detail-section counterfactual-section">
        <CounterfactualAnalysis value={counterfactual} />
      </section>
    </div>
  );
}
