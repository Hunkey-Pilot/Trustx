export type TransactionType =
  | "CASH_IN"
  | "CASH_OUT"
  | "DEBIT"
  | "PAYMENT"
  | "TRANSFER";

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export type ReviewStatus =
  | "OPEN"
  | "UNDER_REVIEW"
  | "DISMISSED"
  | "ESCALATED"
  | "CONFIRMED_SUSPICIOUS";

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

export type SignalSource = "analysis_time" | "legacy_recomputed";

export interface BehavioralSignalsResponse {
  status: "available" | "unavailable";
  source?: SignalSource | null;
  error?: string | null;
  sender: {
    current_amount: number;
    transaction_count_5m: number;
    transaction_count_1h: number;
    transaction_count_24h: number;
    amount_sum_1h: number;
    amount_sum_24h: number;
    historical_average_amount: number | null;
    amount_to_historical_average: number | null;
    historical_max_amount: number | null;
    amount_to_historical_max: number | null;
  } | null;
  recipient: {
    transaction_count: number;
    unique_senders: number;
    amount_sum: number;
    transaction_count_1h: number;
    transaction_count_24h: number;
    unique_senders_1h: number;
    unique_senders_24h: number;
  } | null;
  sender_recipient: {
    pair_transaction_count: number;
    pair_amount_sum: number;
    is_new_recipient_for_sender: boolean;
    minutes_since_previous_pair_transaction: number | null;
  } | null;
}

export interface NetworkAnalysisResponse {
  status: "available" | "unavailable";
  source?: SignalSource | null;
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

export interface InvestigationAction {
  code: string;
  label: string;
  rationale: string;
}

export interface InvestigatorSummary {
  investigator_summary: string;
  key_facts: string[];
  investigation_recommendations: InvestigationAction[];
  disclaimer: string;
}

export interface TransactionAnalysisResponse {
  transaction_id: string;
  fraud_probability: number;
  anomaly_signal: number;
  risk_score: number;
  risk_level: RiskLevel;
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
      risk_level: RiskLevel;
      recommended_action: string;
    };
  };
  behavioral_signals: BehavioralSignalsResponse | null;
  network_signals: NetworkAnalysisResponse | null;
  investigator_summary: InvestigatorSummary | null;
  is_duplicate: boolean;
  model_version: string | null;
  feature_count: number;
}

export interface CounterfactualResponse {
  transaction_id: string;
  original_amount: number;
  original_fraud_probability: number;
  baseline_fraud_probability: number | null;
  baseline_source: "recomputed" | "stored";
  baseline_matches_stored: boolean | null;
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
  interpretation: {
    model_output: string;
    investigation_recommendations: InvestigationAction[];
    disclaimer: string;
  } | null;
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

export interface PersistedTransaction
  extends Omit<TransactionAnalysisResponse, "is_duplicate"> {
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
  review_status: ReviewStatus | null;
}

export interface TransactionListResponse {
  total: number;
  page: number;
  page_size: number;
  returned_items: number;
  items: PersistedTransaction[];
}

export interface ReviewResponse {
  transaction_id: string;
  status: ReviewStatus;
  analyst_note: string | null;
  decision: string | null;
  reviewed_by: string | null;
  reviewed_at: string | null;
  created_at: string;
  updated_at: string;
  model_assessment: {
    risk_level: RiskLevel;
    risk_score: number;
    fraud_probability: number;
    recommended_action: string;
  } | null;
}

export interface ReviewInput {
  status?: ReviewStatus;
  analyst_note?: string | null;
  decision?: string | null;
}

/**
 * All browser requests go to the same-origin proxy route (app/api/trustx), which
 * attaches the API key on the server. The backend base URL is configured through
 * NEXT_PUBLIC_API_BASE_URL on the frontend server (see the route handler).
 */
const PROXY_BASE = "/api/trustx";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function readDetail(response: Response): Promise<string | null> {
  try {
    const body = (await response.json()) as {
      detail?: unknown;
      errors?: Array<{ loc?: string[]; message?: string }>;
    };
    if (typeof body.detail !== "string") return null;
    if (Array.isArray(body.errors) && body.errors.length > 0) {
      const first = body.errors
        .slice(0, 3)
        .map((item) => `${(item.loc ?? []).filter((part) => part !== "body").join(".")}: ${item.message ?? "invalid"}`)
        .join("; ");
      return `${body.detail} (${first})`;
    }
    return body.detail;
  } catch {
    return null;
  }
}

async function request<T>(
  path: string,
  errorMessage: string,
  init?: RequestInit,
  reviewer?: string,
): Promise<T> {
  let response: Response;
  const headers: Record<string, string> = {};
  if (init?.body) headers["Content-Type"] = "application/json";
  if (reviewer && reviewer.trim()) headers["X-TrustX-User"] = reviewer.trim();
  try {
    response = await fetch(`${PROXY_BASE}${path}`, {
      cache: "no-store",
      ...init,
      headers,
    });
  } catch {
    throw new ApiError(errorMessage, 0);
  }
  if (!response.ok) {
    const detail = await readDetail(response);
    if (response.status === 404) {
      throw new ApiError(detail ?? "Not found.", 404);
    }
    throw new ApiError(detail ?? errorMessage, response.status);
  }
  return (await response.json()) as T;
}

const DASHBOARD_ERROR =
  "Unable to load dashboard. Please check that the FastAPI server is running.";

export function getTransactionSummary(): Promise<TransactionSummaryResponse> {
  return request("/api/v1/transactions/summary", DASHBOARD_ERROR);
}

export function getRiskSummary(): Promise<TransactionSummaryResponse> {
  return request("/api/v1/risk/summary", DASHBOARD_ERROR);
}

export function getTransactions(pageSize = 10): Promise<TransactionListResponse> {
  const query = new URLSearchParams({ page: "1", page_size: String(pageSize) });
  return request(`/api/v1/transactions?${query.toString()}`, DASHBOARD_ERROR);
}

/** HIGH/CRITICAL transactions without a closed review, highest risk first. */
export function getReviewQueue(pageSize = 10): Promise<TransactionListResponse> {
  const query = new URLSearchParams({
    page: "1",
    page_size: String(pageSize),
    review_queue: "true",
    sort_by: "risk_priority",
    sort_order: "desc",
  });
  return request(`/api/v1/transactions?${query.toString()}`, DASHBOARD_ERROR);
}

export function getTransaction(transactionId: string): Promise<PersistedTransaction> {
  return request(
    `/api/v1/transactions/${encodeURIComponent(transactionId)}`,
    "Unable to load transaction. Please check that the FastAPI server is running.",
  );
}

export function analyzeTransaction(
  payload: TransactionAnalysisRequest,
): Promise<TransactionAnalysisResponse> {
  return request(
    "/api/v1/transactions/analyze",
    "Unable to analyze transaction. Please check that the FastAPI server is running.",
    { method: "POST", body: JSON.stringify(payload) },
  );
}

export function getTransactionCounterfactual(
  transactionId: string,
): Promise<CounterfactualResponse> {
  return request(
    `/api/v1/transactions/${encodeURIComponent(transactionId)}/counterfactual`,
    "Counterfactual explanation is not available.",
  );
}

export function getNetworkAnalysis(
  transactionId: string,
): Promise<NetworkAnalysisResponse> {
  return request(
    `/api/v1/transactions/${encodeURIComponent(transactionId)}/network`,
    "Network analysis is not available.",
  );
}

export async function getReview(transactionId: string): Promise<ReviewResponse | null> {
  try {
    return await request<ReviewResponse>(
      `/api/v1/reviews/${encodeURIComponent(transactionId)}`,
      "Unable to load the analyst review.",
    );
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }
}

export function createReview(
  transactionId: string,
  input: ReviewInput,
  reviewer?: string,
): Promise<ReviewResponse> {
  return request(
    `/api/v1/reviews/${encodeURIComponent(transactionId)}`,
    "Unable to save the analyst review.",
    { method: "POST", body: JSON.stringify(input) },
    reviewer,
  );
}

export function updateReview(
  transactionId: string,
  input: ReviewInput,
  reviewer?: string,
): Promise<ReviewResponse> {
  return request(
    `/api/v1/reviews/${encodeURIComponent(transactionId)}`,
    "Unable to save the analyst review.",
    { method: "PATCH", body: JSON.stringify(input) },
    reviewer,
  );
}
