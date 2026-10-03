import type {
  CounterfactualResponse,
  SignalPayload,
  TransactionAnalysisResponse,
} from "@/lib/api";
import CounterfactualAnalysis from "@/components/CounterfactualAnalysis";

interface AnalysisResultProps {
  result: TransactionAnalysisResponse | null;
  counterfactual: CounterfactualResponse | null;
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US", {
    maximumSignificantDigits: 7,
  }).format(value);
}

function SignalSection({
  title,
  signal,
}: {
  title: string;
  signal: SignalPayload | null;
}) {
  const available = signal?.status === "available";
  return (
    <section className="detail-section">
      <div className="section-heading">
        <h3>{title}</h3>
        <span className={available ? "availability available" : "availability"}>
          {available ? "Available" : "Not available"}
        </span>
      </div>
      {available ? (
        <pre className="signal-data">{JSON.stringify(signal, null, 2)}</pre>
      ) : (
        <p className="muted-copy">Not available</p>
      )}
    </section>
  );
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
      <section className="summary-section">
        <div className="result-title-row">
          <div>
            <p className="eyebrow">Analysis result</p>
            <h2>Model assessment</h2>
          </div>
          <span className={`risk-tag risk-${result.risk_level.toLowerCase()}`}>
            {result.risk_level}
          </span>
        </div>

        <div className="metric-grid">
          <div className="metric metric-primary">
            <span>Fraud probability</span>
            <strong>{formatNumber(result.fraud_probability)}</strong>
          </div>
          <div className="metric">
            <span>Risk score</span>
            <strong>{formatNumber(result.risk_score)}</strong>
          </div>
          <div className="metric">
            <span>Anomaly signal</span>
            <strong>{formatNumber(result.anomaly_signal)}</strong>
            <small>Raw Isolation Forest decision value</small>
          </div>
          <div className="metric">
            <span>Recommended action</span>
            <strong className="action-value">{result.recommended_action}</strong>
          </div>
        </div>
      </section>

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

      <SignalSection title="Behavioral signals" signal={result.behavioral_signals} />
      <SignalSection title="Network signals" signal={result.network_signals} />

      <section className="detail-section counterfactual-section">
        <CounterfactualAnalysis value={counterfactual} />
      </section>
    </div>
  );
}