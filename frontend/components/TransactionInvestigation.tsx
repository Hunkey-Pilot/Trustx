"use client";

import { useEffect, useState } from "react";
import SiteHeader from "@/components/SiteHeader";
import CounterfactualAnalysis from "@/components/CounterfactualAnalysis";
import {
  getNetworkAnalysis,
  getTransaction,
  getTransactionCounterfactual,
  type CounterfactualResponse,
  type NetworkAnalysisResponse,
  type PersistedTransaction,
  type SignalPayload,
} from "@/lib/api";

function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US", {
    maximumSignificantDigits: 8,
  }).format(value);
}

function riskClass(level: string): string {
  return `risk-tag risk-${level.toLowerCase()}`;
}

function SignalBlock({ title, value }: { title: string; value: SignalPayload | null }) {
  return (
    <section className="investigation-section">
      <div className="section-title-row">
        <h2>{title}</h2>
        <span className={value?.status === "available" ? "availability available" : "availability"}>
          {value?.status === "available" ? "Available" : "Not available"}
        </span>
      </div>
      {value?.status === "available" ? (
        <pre className="investigation-json">{JSON.stringify(value, null, 2)}</pre>
      ) : (
        <p className="muted-copy">Not available</p>
      )}
    </section>
  );
}

function NetworkAnalysisBlock({ value }: { value: NetworkAnalysisResponse | null }) {
  const detectedPatterns = value?.patterns.filter((pattern) => pattern.detected) ?? [];

  return (
    <section className="investigation-section">
      <div className="section-title-row">
        <div>
          <p className="eyebrow">Relationship activity</p>
          <h2>Network Analysis</h2>
        </div>
        <span className={value?.status === "available" ? "availability available" : "availability"}>
          {value?.status === "available" ? "Available" : "Not available"}
        </span>
      </div>
      {value?.status !== "available" || !value.sender || !value.recipient || !value.relationship ? (
        <p className="muted-copy">Not available</p>
      ) : (
        <>
          <div className="network-analysis-grid">
            <div className="network-group">
              <h3>Sender Activity</h3>
              <dl>
                <div><dt>Outgoing Transactions</dt><dd>{value.sender.outgoing_transaction_count}</dd></div>
                <div><dt>Unique Recipients</dt><dd>{value.sender.unique_recipient_count}</dd></div>
                <div><dt>Total Outgoing Amount</dt><dd>{formatNumber(value.sender.total_outgoing_amount)}</dd></div>
              </dl>
            </div>
            <div className="network-group">
              <h3>Recipient Activity</h3>
              <dl>
                <div><dt>Incoming Transactions</dt><dd>{value.recipient.incoming_transaction_count}</dd></div>
                <div><dt>Unique Senders</dt><dd>{value.recipient.unique_sender_count}</dd></div>
                <div><dt>Total Incoming Amount</dt><dd>{formatNumber(value.recipient.total_incoming_amount)}</dd></div>
              </dl>
            </div>
            <div className="network-group relationship-group">
              <h3>Relationship</h3>
              <dl>
                <div><dt>Previous Transactions</dt><dd>{value.relationship.previous_transaction_count}</dd></div>
                <div><dt>Previous Amount</dt><dd>{formatNumber(value.relationship.previous_transaction_amount)}</dd></div>
                <div><dt>Relationship Status</dt><dd>{value.relationship.relationship_status.replaceAll("_", " ")}</dd></div>
              </dl>
            </div>
          </div>
          <div className="network-patterns">
            <h3>Network Patterns</h3>
            {detectedPatterns.length ? (
              <ul>
                {detectedPatterns.map((pattern) => (
                  <li key={pattern.pattern}>
                    {pattern.pattern === "HIGH_RECIPIENT_CONNECTIVITY"
                      ? "High recipient connectivity observed"
                      : "High sender connectivity observed"}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted-copy">No notable network pattern detected</p>
            )}
          </div>
        </>
      )}
    </section>
  );
}

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
          getNetworkAnalysis(transactionId).catch(() => null),
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
                <h2>Transaction information</h2>
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
                <dd>{formatNumber(transaction.amount)}</dd>
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
                <dt>Step</dt>
                <dd>{transaction.step}</dd>
              </div>
              <div>
                <dt>Created at</dt>
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
                <span>Fraud probability</span>
                <strong>{formatNumber(transaction.fraud_probability)}</strong>
              </div>
              <div className="investigation-metric">
                <span>Risk score</span>
                <strong>{formatNumber(transaction.risk_score)}</strong>
              </div>
              <div className="investigation-metric action-metric">
                <span>Recommended action</span>
                <strong>{transaction.recommended_action}</strong>
              </div>
            </div>
          </section>

          <section className="investigation-section anomaly-section">
            <div>
              <p className="eyebrow">Isolation Forest</p>
              <h2>Anomaly signal</h2>
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

          <SignalBlock title="Behavioral signals" value={transaction.behavioral_signals} />
          <NetworkAnalysisBlock value={networkAnalysis} />
          <section className="investigation-section">
            <CounterfactualAnalysis value={counterfactual} />
          </section>
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