"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import SiteHeader from "@/components/SiteHeader";
import {
  getRiskSummary,
  getTransactionSummary,
  getTransactions,
  type TransactionListResponse,
  type TransactionSummaryResponse,
} from "@/lib/api";

function formatNumber(value: number | null): string {
  if (value === null) return "Not available";
  return new Intl.NumberFormat("en-US", {
    maximumSignificantDigits: 7,
  }).format(value);
}

function riskClass(level: string): string {
  return `risk-tag risk-${level.toLowerCase()}`;
}

export default function DashboardPage() {
  const [transactionSummary, setTransactionSummary] =
    useState<TransactionSummaryResponse | null>(null);
  const [riskSummary, setRiskSummary] =
    useState<TransactionSummaryResponse | null>(null);
  const [transactions, setTransactions] =
    useState<TransactionListResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadDashboard() {
      try {
        const [transactionData, riskData, recentData] = await Promise.all([
          getTransactionSummary(),
          getRiskSummary(),
          getTransactions(),
        ]);
        if (!cancelled) {
          setTransactionSummary(transactionData);
          setRiskSummary(riskData);
          setTransactions(recentData);
        }
      } catch {
        if (!cancelled) {
          setError(
            "Unable to load dashboard. Please check that the FastAPI server is running.",
          );
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    void loadDashboard();
    return () => {
      cancelled = true;
    };
  }, []);

  const riskCounts = riskSummary?.count_by_risk_level ?? {};

  return (
    <main className="workspace-shell dashboard-shell">
      <SiteHeader current="dashboard" />
      <section className="page-intro dashboard-intro">
        <div>
          <p className="eyebrow">Overview</p>
          <h1>TrustX</h1>
        </div>
        <p className="intro-aside">AI Trust &amp; Risk Intelligence Platform</p>
      </section>

      {error ? (
        <section className="dashboard-state error-state" role="alert">
          <span className="error-symbol" aria-hidden="true">!</span>
          <p>{error}</p>
        </section>
      ) : (
        <>
          <section className="dashboard-summary-section" aria-label="Transaction summary">
            <div className="section-title-row">
              <div>
                <p className="eyebrow">Portfolio</p>
                <h2>Risk summary</h2>
              </div>
              {isLoading && <span className="inline-loading">Loading summary...</span>}
            </div>
            <div className="dashboard-summary-grid">
              <article className="summary-card total-card">
                <span>Total transactions</span>
                <strong>
                  {transactionSummary?.total_analyzed_transactions.toLocaleString() ?? "-"}
                </strong>
              </article>
              {(["LOW", "MEDIUM", "HIGH", "CRITICAL"] as const).map((level) => (
                <article
                  className={`summary-card summary-risk-${level.toLowerCase()}`}
                  key={level}
                >
                  <span>{level.charAt(0) + level.slice(1).toLowerCase()} risk</span>
                  <strong>{(riskCounts[level] ?? 0).toLocaleString()}</strong>
                </article>
              ))}
              <article className="summary-card">
                <span>Average fraud probability</span>
                <strong>{formatNumber(transactionSummary?.average_fraud_probability ?? null)}</strong>
              </article>
              <article className="summary-card">
                <span>Average risk score</span>
                <strong>{formatNumber(riskSummary?.average_risk_score ?? null)}</strong>
              </article>
            </div>
          </section>

          <section className="transactions-section" id="transactions">
            <div className="section-title-row">
              <div>
                <p className="eyebrow">Latest activity</p>
                <h2>Recent transactions</h2>
              </div>
              {transactions && (
                <span className="inline-loading">
                  {transactions.total.toLocaleString()} total
                </span>
              )}
            </div>

            {isLoading ? (
              <div className="table-state" role="status">Loading transactions...</div>
            ) : transactions?.items.length ? (
              <div className="table-scroll">
                <table className="transactions-table">
                  <thead>
                    <tr>
                      <th>Transaction ID</th>
                      <th>Type</th>
                      <th>Amount</th>
                      <th>Fraud probability</th>
                      <th>Risk score</th>
                      <th>Risk level</th>
                      <th>Recommended action</th>
                      <th>Created at</th>
                      <th aria-label="Open transaction" />
                    </tr>
                  </thead>
                  <tbody>
                    {transactions.items.map((transaction) => (
                      <tr key={transaction.transaction_id}>
                        <td className="transaction-id">{transaction.transaction_id}</td>
                        <td>{transaction.type}</td>
                        <td className="numeric-cell">{formatNumber(transaction.amount)}</td>
                        <td className="numeric-cell">{formatNumber(transaction.fraud_probability)}</td>
                        <td className="numeric-cell">{formatNumber(transaction.risk_score)}</td>
                        <td>
                          <span className={riskClass(transaction.risk_level)}>
                            {transaction.risk_level}
                          </span>
                        </td>
                        <td className="action-cell">{transaction.recommended_action}</td>
                        <td className="date-cell">
                          {new Date(transaction.created_at).toLocaleString()}
                        </td>
                        <td>
                          <Link
                            className="view-link"
                            href={`/transactions/${encodeURIComponent(transaction.transaction_id)}`}
                          >
                            View
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="table-state">No transactions found.</div>
            )}
          </section>
        </>
      )}

      <footer className="page-footer">
        <span>TrustX</span>
        <span>Model assessments are signals, not determinations.</span>
      </footer>
    </main>
  );
}
