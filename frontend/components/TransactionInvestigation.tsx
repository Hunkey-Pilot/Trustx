"use client";

import { useEffect, useState } from "react";
import SiteHeader from "@/components/SiteHeader";
import AnalystReview from "@/components/AnalystReview";
import BehavioralSignalsPanel from "@/components/BehavioralSignalsPanel";
import CounterfactualAnalysis from "@/components/CounterfactualAnalysis";
import InvestigatorSummaryPanel from "@/components/InvestigatorSummaryPanel";
import NetworkAnalysisPanel from "@/components/NetworkAnalysisPanel";
import {
  getNetworkAnalysis,
  getTransaction,
  getTransactionCounterfactual,
  type CounterfactualResponse,
  type NetworkAnalysisResponse,
  type PersistedTransaction,
} from "@/lib/api";
import { MODEL_SCORE_DISCLAIMER, formatNumber, riskClass } from "@/lib/format";

export default function TransactionInvestigation({
  transactionId,
}: {
  transactionId: string;
}) {
  const [transaction, setTransaction] = useState<PersistedTransaction | null>(null);
  const [counterfactual, setCounterfactual] =
    useState<CounterfactualResponse | null>(null);
  const [networkAnalysis, setNetworkAnalysis] =
    useState<NetworkAnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadTransaction() {
      try {
        const persisted = await getTransaction(transactionId);
        if (cancelled) return;
        setTransaction(persisted);
        const [network, counterfactualData] = await Promise.all([
          persisted.network_signals
            ? Promise.resolve(persisted.network_signals)
            : getNetworkAnalysis(transactionId).catch(() => null),
          getTransactionCounterfactual(transactionId).catch(() => null),
        ]);
        if (!cancelled) {
          setNetworkAnalysis(network);
          setCounterfactual(counterfactualData);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : "Unable to load transaction. Please check that the FastAPI server is running.",
          );
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    void loadTransaction();
    return () => {
      cancelled = true;
    };
  }, [transactionId]);

  return (
    <main className="workspace-shell investigation-shell">
      <SiteHeader current="transactions" />
      <section className="page-intro investigation-intro">
        <div>
          <p className="eyebrow">Transaction investigation</p>
          <h1>Transaction detail</h1>
        </div>
        <p className="intro-aside investigation-id">{transactionId}</p>
      </section>

      {isLoading ? (
        <section className="investigation-state" role="status">
          <span className="loading-rule" aria-hidden="true" />
          <p>Loading transaction...</p>
        </section>
      ) : error ? (
        <section className="investigation-state error-state" role="alert">
          <span className="error-symbol" aria-hidden="true">!</span>
          <p>{error}</p>
        </section>
      ) : transaction ? (
        <>
          <section className="investigation-section transaction-info-section">
            <div className="section-title-row">
              <div>
                <p className="eyebrow">Record</p>
                <h2>Transaction overview</h2>
              </div>
              <span className="transaction-type-tag">{transaction.type}</span>
            </div>
            <dl className="transaction-info-grid">
              <div>
                <dt>Transaction ID</dt>
                <dd className="mono-value">{transaction.transaction_id}</dd>
              </div>
              <div>
                <dt>Amount</dt>
                <dd>{formatNumber(transaction.amount, 8)}</dd>
              </div>
              <div>
                <dt>Sender</dt>
                <dd>{transaction.nameOrig}</dd>
              </div>
              <div>
                <dt>Recipient</dt>
                <dd>{transaction.nameDest}</dd>
              </div>
              <div>
                <dt>Sender balance</dt>
                <dd>
                  {formatNumber(transaction.oldbalanceOrg, 8)} → {formatNumber(transaction.newbalanceOrig, 8)}
                </dd>
              </div>
              <div>
                <dt>Recipient balance</dt>
                <dd>
                  {formatNumber(transaction.oldbalanceDest, 8)} → {formatNumber(transaction.newbalanceDest, 8)}
                </dd>
              </div>
              <div>
                <dt>Step</dt>
                <dd>{transaction.step}</dd>
              </div>
              <div>
                <dt>Analyzed at</dt>
                <dd>{new Date(transaction.created_at).toLocaleString()}</dd>
              </div>
            </dl>
          </section>

          <section className="investigation-section">
            <div className="section-title-row">
              <div>
                <p className="eyebrow">Assessment</p>
                <h2>Risk assessment</h2>
              </div>
              <span className={riskClass(transaction.risk_level)}>{transaction.risk_level}</span>
            </div>
            <div className="investigation-metrics">
              <div className="investigation-metric">
                <span>Model score</span>
                <strong>{formatNumber(transaction.fraud_probability)}</strong>
                <small>Uncalibrated model output</small>
              </div>
              <div className="investigation-metric">
                <span>Risk score</span>
                <strong>{formatNumber(transaction.risk_score)}</strong>
                <small>Model score x 100</small>
              </div>
              <div className="investigation-metric action-metric">
                <span>Recommended action</span>
                <strong>{transaction.recommended_action}</strong>
              </div>
            </div>
            <p className="score-disclaimer">{MODEL_SCORE_DISCLAIMER}</p>
          </section>

          <InvestigatorSummaryPanel value={transaction.investigator_summary} />

          <section className="investigation-section anomaly-section">
            <div>
              <p className="eyebrow">Isolation Forest</p>
              <h2>Anomaly signal</h2>
              <p className="source-note">
                Raw decision value; below 0 means the transaction looks more unusual than the data the
                model was trained on. It is shown for context and is not part of the risk score.
              </p>
            </div>
            <strong>{formatNumber(transaction.anomaly_signal)}</strong>
          </section>

          <section className="investigation-section">
            <div className="section-title-row">
              <div>
                <p className="eyebrow">Model attribution</p>
                <h2>SHAP explanation</h2>
              </div>
              <span className="availability">{transaction.explanation.status}</span>
            </div>
            {transaction.explanation.top_features.length ? (
              <div className="investigation-shap-list">
                {transaction.explanation.top_features.map((feature) => (
                  <article className="investigation-shap-row" key={feature.feature}>
                    <strong>{feature.feature}</strong>
                    <span>Value <b>{formatNumber(feature.value)}</b></span>
                    <span>SHAP <b>{formatNumber(feature.shap_value)}</b></span>
                    <span className="shap-direction">
                      {feature.direction.replaceAll("_", " ")}
                    </span>
                  </article>
                ))}
              </div>
            ) : (
              <p className="muted-copy">Not available</p>
            )}
            <p className="source-note">SHAP describes how the model used its inputs; it is not causal proof.</p>
          </section>

          <section className="investigation-section">
            <div className="section-title-row">
              <div>
                <p className="eyebrow">Supporting signals</p>
                <h2>Evidence</h2>
              </div>
              <span className="availability">{transaction.evidence.status}</span>
            </div>
            {transaction.evidence.items.length ? (
              <ul className="investigation-evidence-list">
                {transaction.evidence.items.map((item, index) => (
                  <li key={`${item.category}-${item.type}-${index}`}>
                    <div>
                      <span className="evidence-category">{item.category.replaceAll("_", " ")}</span>
                      <strong>{item.type.replaceAll("_", " ")}</strong>
                    </div>
                    <p>{item.description}</p>
                    {item.value !== null && <span className="evidence-value">{formatNumber(item.value)}</span>}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted-copy">Not available</p>
            )}
          </section>

          <BehavioralSignalsPanel value={transaction.behavioral_signals} />
          <NetworkAnalysisPanel value={networkAnalysis} />
          <section className="investigation-section">
            <CounterfactualAnalysis value={counterfactual} />
          </section>

          <AnalystReview
            transactionId={transaction.transaction_id}
            riskLevel={transaction.risk_level}
          />
        </>
      ) : (
        <section className="investigation-state">
          <p>Transaction not found.</p>
        </section>
      )}

      <footer className="page-footer">
        <span>TrustX</span>
        <span>Model assessments are signals, not determinations.</span>
      </footer>
    </main>
  );
}
