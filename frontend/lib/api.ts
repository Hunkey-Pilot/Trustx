export type TransactionType =
  | "CASH_IN"
  | "CASH_OUT"
  | "DEBIT"
  | "PAYMENT"
  | "TRANSFER";

export interface HistoricalTransaction {
  step: number;
  amount: number;
  nameOrig: string;
}

export interface TransactionAnalysisRequest {
  step: number;
  type: TransactionType;
  amount: number;
  nameOrig: string;
  oldbalanceOrg: number;
  newbalanceOrig: number;
  nameDest: string;
  oldbalanceDest: number;
  newbalanceDest: number;
  historical_transactions: HistoricalTransaction[];
}

export interface ShapFeatureContribution {
  feature: string;
  value: number;
  shap_value: number;
  direction: "increases_fraud_prediction" | "decreases_fraud_prediction" | "neutral";
}

export interface EvidenceItem {
  category:
    | "MODEL_SIGNAL"
    | "ANOMALY_SIGNAL"
    | "MODEL_ATTRIBUTION"
    | "BEHAVIORAL_SIGNAL"
    | "NETWORK_SIGNAL";
  type: string;
  value: number | null;
  description: string;
  feature: string | null;
  shap_value: number | null;
  direction: string | null;
}

export interface SignalPayload {
  status: "available" | "unavailable";
  error?: string | null;
  [key: string]: unknown;
}

export interface TransactionAnalysisResponse {
  transaction_id: string;
  fraud_probability: number;
  anomaly_signal: number;
  risk_score: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  recommended_action: string;
  explanation: {
    status: "available" | "unavailable";
    method: "SHAP";
    type: "model_attribution";
    output_space: string | null;
    base_value: number | null;
    top_features: ShapFeatureContribution[];
    error: string | null;
  };
  evidence: {
    status: "available" | "partial";
    items: EvidenceItem[];
    risk_assessment: {
      risk_score: number;
      risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
      recommended_action: string;
    };
  };
  behavioral_signals: SignalPayload | null;
  network_signals: SignalPayload | null;
  model_version: string | null;
  feature_count: number;
}

export interface CounterfactualResponse {
  transaction_id: string;
  original_amount: number;
  original_fraud_probability: number;
  counterfactuals: Array<{
    amount: number;
    amount_ratio: number;
    fraud_probability: number | null;
    probability_change: number | null;
    valid: boolean;
  }>;
  best_counterfactual: {
    amount: number;
    amount_ratio: number;
    fraud_probability: number;
    probability_reduction: number;
    description: string;
  } | null;
}

export interface NetworkAnalysisResponse {
  status: "available" | "unavailable";
  transaction_id: string | null;
  sender: {
    outgoing_transaction_count: number;
    unique_recipient_count: number;
    total_outgoing_amount: number;
  } | null;
  recipient: {
    incoming_transaction_count: number;
    unique_sender_count: number;
    total_incoming_amount: number;
  } | null;
  relationship: {
    previous_transaction_count: number;
    previous_transaction_amount: number;
    relationship_status: "NEW_RELATIONSHIP" | "EXISTING_RELATIONSHIP";
  } | null;
  patterns: Array<{
    pattern: "HIGH_RECIPIENT_CONNECTIVITY" | "HIGH_SENDER_CONNECTIVITY";
    detected: boolean;
  }>;
  error: string | null;
}

export interface TransactionSummaryResponse {
  total_analyzed_transactions: number;
  count_by_risk_level: Record<string, number>;
  count_by_recommended_action: Record<string, number>;
  count_by_transaction_type: Record<string, number>;
  average_fraud_probability: number | null;
  maximum_fraud_probability: number | null;
  average_risk_score: number | null;
  maximum_risk_score: number | null;
}

export interface PersistedTransaction extends TransactionAnalysisResponse {
  id: number;
  created_at: string;
  step: number;
  type: TransactionType;
  amount: number;
  nameOrig: string;
  oldbalanceOrg: number;
  newbalanceOrig: number;
  nameDest: string;
  oldbalanceDest: number;
  newbalanceDest: number;
  historical_transactions: HistoricalTransaction[];
}

export interface TransactionListResponse {
  total: number;
  page: number;
  page_size: number;
  returned_items: number;
  items: PersistedTransaction[];
}

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/+$/, "");

async function getJson<T>(path: string, errorMessage: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  } catch {
    throw new Error(errorMessage);
  }
  if (!response.ok) {
    throw new Error(response.status === 404 ? "Transaction not found." : errorMessage);
  }
  return (await response.json()) as T;
}

const DASHBOARD_ERROR =
  "Unable to load dashboard. Please check that the FastAPI server is running.";

export function getTransactionSummary(): Promise<TransactionSummaryResponse> {
  return getJson("/api/v1/transactions/summary", DASHBOARD_ERROR);
}

export function getRiskSummary(): Promise<TransactionSummaryResponse> {
  return getJson("/api/v1/risk/summary", DASHBOARD_ERROR);
}

export function getTransactions(pageSize = 10): Promise<TransactionListResponse> {
  const query = new URLSearchParams({ page: "1", page_size: String(pageSize) });
  return getJson(`/api/v1/transactions?${query.toString()}`, DASHBOARD_ERROR);
}

export function getTransaction(transactionId: string): Promise<PersistedTransaction> {
  return getJson(
    `/api/v1/transactions/${encodeURIComponent(transactionId)}`,
    "Unable to load transaction. Please check that the FastAPI server is running.",
  );
}

export async function analyzeTransaction(
  payload: TransactionAnalysisRequest,
): Promise<TransactionAnalysisResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/transactions/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error(
      "Unable to analyze transaction. Please check that the FastAPI server is running.",
    );
  }

  if (!response.ok) {
    throw new Error(
      "Unable to analyze transaction. Please check that the FastAPI server is running.",
    );
  }

  return (await response.json()) as TransactionAnalysisResponse;
}

export async function getTransactionCounterfactual(
  transactionId: string,
): Promise<CounterfactualResponse> {
  return getJson(
    `/api/v1/transactions/${encodeURIComponent(transactionId)}/counterfactual`,
    "Counterfactual explanation is not available.",
  );
}

export function getNetworkAnalysis(
  transactionId: string,
): Promise<NetworkAnalysisResponse> {
  return getJson(
    `/api/v1/transactions/${encodeURIComponent(transactionId)}/network`,
    "Network analysis is not available.",
  );
}