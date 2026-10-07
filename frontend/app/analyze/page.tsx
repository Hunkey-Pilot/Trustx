"use client";

import { useState } from "react";
import AnalysisResult from "@/components/AnalysisResult";
import SiteHeader from "@/components/SiteHeader";
import TransactionForm from "@/components/TransactionForm";
import {
  ApiError,
  analyzeTransaction,
  getTransactionCounterfactual,
  type CounterfactualResponse,
  type TransactionAnalysisRequest,
  type TransactionAnalysisResponse,
} from "@/lib/api";

export default function AnalyzePage() {
  const [result, setResult] = useState<TransactionAnalysisResponse | null>(null);
  const [counterfactual, setCounterfactual] =
    useState<CounterfactualResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyze(payload: TransactionAnalysisRequest) {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setCounterfactual(null);

    try {
      const analysis = await analyzeTransaction(payload);
      setResult(analysis);
      try {
        setCounterfactual(await getTransactionCounterfactual(analysis.transaction_id));
      } catch {
        setCounterfactual(null);
      }
    } catch (analysisError) {
      setError(
        analysisError instanceof ApiError
          ? analysisError.message
          : "Unable to analyze transaction. Please check that the FastAPI server is running.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="workspace-shell">
      <SiteHeader current="analysis" />
      <section className="page-intro">
        <div>
          <p className="eyebrow">Trust &amp; risk intelligence</p>
          <h1>Analyze a transaction.</h1>
        </div>
        <p className="intro-aside">Model outputs and supporting signals, in one view.</p>
      </section>

      <div className="workspace-grid">
        <section className="input-panel" aria-label="Transaction input">
          <TransactionForm isLoading={isLoading} onAnalyze={handleAnalyze} />
        </section>
        <section className="result-panel" aria-label="Analysis result">
          <div className="result-panel-heading">
            <p className="eyebrow">Output</p>
            <span className="result-index">
              {result ? result.transaction_id : "Awaiting analysis"}
            </span>
          </div>
          {isLoading ? (
            <div className="loading-state" role="status" aria-live="polite">
              <span className="loading-rule" aria-hidden="true" />
              <p>Analyzing transaction...</p>
            </div>
          ) : error ? (
            <div className="error-state" role="alert">
              <span className="error-symbol" aria-hidden="true">!</span>
              <p>{error}</p>
            </div>
          ) : (
            <AnalysisResult result={result} counterfactual={counterfactual} />
          )}
        </section>
      </div>

      <footer className="page-footer">
        <span>TrustX</span>
        <span>Model assessments are signals, not determinations.</span>
      </footer>
    </main>
  );
}