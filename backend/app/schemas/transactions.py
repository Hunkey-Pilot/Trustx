from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


# Input limits. PaySim steps run 1..743 and amounts reach ~9.2e7; the limits below
# leave generous headroom while rejecting absurd values.
MAX_STEP = 100_000
MAX_MONEY = 1e12
MAX_ID_LENGTH = 100
MAX_HISTORY_ITEMS = 500
ACCOUNT_ID_PATTERN = r"^[A-Za-z0-9_.:\-]+$"
TRANSACTION_ID_PATTERN = r"^[A-Za-z0-9_.:\-]+$"
MAX_TRANSACTION_ID_LENGTH = 128

AccountId = Annotated[
    str, Field(min_length=1, max_length=MAX_ID_LENGTH, pattern=ACCOUNT_ID_PATTERN)
]


TransactionType = Literal[
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


class HistoricalTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0, le=MAX_STEP)
    amount: FiniteFloat = Field(ge=0, le=MAX_MONEY)
    nameOrig: AccountId


class StoredHistoricalTransaction(BaseModel):
    """Lenient read model: rows stored before input limits existed must stay readable."""

    step: int
    amount: FiniteFloat
    nameOrig: str


class TransactionAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0, le=MAX_STEP)
    type: TransactionType
    amount: FiniteFloat = Field(ge=0, le=MAX_MONEY)
    nameOrig: AccountId
    oldbalanceOrg: FiniteFloat = Field(ge=0, le=MAX_MONEY)
    newbalanceOrig: FiniteFloat = Field(ge=0, le=MAX_MONEY)
    nameDest: AccountId
    oldbalanceDest: FiniteFloat = Field(ge=0, le=MAX_MONEY)
    newbalanceDest: FiniteFloat = Field(ge=0, le=MAX_MONEY)
    historical_transactions: list[HistoricalTransaction] = Field(
        default_factory=list, max_length=MAX_HISTORY_ITEMS
    )


class TransactionAnalyzeResponse(BaseModel):
    transaction_id: str
    fraud_probability: FiniteFloat
    anomaly_signal: FiniteFloat
    risk_score: FiniteFloat = Field(ge=0, le=100)
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    recommended_action: str
    explanation: "ShapExplanation"
    evidence: "EvidenceResponse"
    behavioral_signals: "BehavioralSignals | None" = None
    network_signals: "NetworkSignals | None" = None
    investigator_summary: "InvestigatorSummary | None" = None
    is_duplicate: bool = False
    model_version: str | None
    feature_count: int


class ShapFeatureContribution(BaseModel):
    feature: str
    value: FiniteFloat
    shap_value: FiniteFloat
    direction: Literal[
        "increases_fraud_prediction",
        "decreases_fraud_prediction",
        "neutral",
    ]


class ShapExplanation(BaseModel):
    status: Literal["available", "unavailable"]
    method: Literal["SHAP"]
    type: Literal["model_attribution"]
    output_space: str | None
    base_value: FiniteFloat | None
    top_features: list[ShapFeatureContribution]
    error: str | None = None


class EvidenceItem(BaseModel):
    category: Literal[
        "MODEL_SIGNAL",
        "ANOMALY_SIGNAL",
        "MODEL_ATTRIBUTION",
        "BEHAVIORAL_SIGNAL",
        "NETWORK_SIGNAL",
    ]
    type: str
    value: FiniteFloat | None
    description: str
    feature: str | None = None
    shap_value: FiniteFloat | None = None
    direction: str | None = None


class RiskAssessment(BaseModel):
    risk_score: FiniteFloat = Field(ge=0, le=100)
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    recommended_action: str


class EvidenceResponse(BaseModel):
    status: Literal["available", "partial"]
    items: list[EvidenceItem]
    risk_assessment: RiskAssessment


class PersistedTransactionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    transaction_id: str
    step: int
    type: TransactionType
    amount: FiniteFloat
    nameOrig: str
    oldbalanceOrg: FiniteFloat
    newbalanceOrig: FiniteFloat
    nameDest: str
    oldbalanceDest: FiniteFloat
    newbalanceDest: FiniteFloat
    historical_transactions: list[StoredHistoricalTransaction]
    fraud_probability: FiniteFloat
    anomaly_signal: FiniteFloat
    risk_score: FiniteFloat = Field(ge=0, le=100)
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    recommended_action: str
    explanation: ShapExplanation
    evidence: EvidenceResponse
    behavioral_signals: BehavioralSignals
    network_signals: NetworkSignals | None = None
    investigator_summary: "InvestigatorSummary | None" = None
    review_status: str | None = None
    model_version: str | None
    feature_count: int


class TransactionListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    returned_items: int
    items: list[PersistedTransactionResponse]


class TransactionSummaryResponse(BaseModel):
    total_analyzed_transactions: int
    count_by_risk_level: dict[str, int]
    count_by_recommended_action: dict[str, int]
    count_by_transaction_type: dict[str, int]
    average_fraud_probability: FiniteFloat | None
    maximum_fraud_probability: FiniteFloat | None
    average_risk_score: FiniteFloat | None
    maximum_risk_score: FiniteFloat | None


class SenderBehaviorSignals(BaseModel):
    current_amount: FiniteFloat
    transaction_count_5m: int
    transaction_count_1h: int
    transaction_count_24h: int
    amount_sum_1h: FiniteFloat
    amount_sum_24h: FiniteFloat
    historical_average_amount: FiniteFloat | None
    amount_to_historical_average: FiniteFloat | None
    historical_max_amount: FiniteFloat | None
    amount_to_historical_max: FiniteFloat | None


class RecipientBehaviorSignals(BaseModel):
    transaction_count: int
    unique_senders: int
    amount_sum: FiniteFloat
    transaction_count_1h: int
    transaction_count_24h: int
    unique_senders_1h: int
    unique_senders_24h: int


class PairBehaviorSignals(BaseModel):
    pair_transaction_count: int
    pair_amount_sum: FiniteFloat
    is_new_recipient_for_sender: bool
    minutes_since_previous_pair_transaction: FiniteFloat | None


SignalSource = Literal["analysis_time", "legacy_recomputed"]


class BehavioralSignals(BaseModel):
    status: Literal["available", "unavailable"]
    source: SignalSource | None = None
    sender: SenderBehaviorSignals | None = None
    recipient: RecipientBehaviorSignals | None = None
    sender_recipient: PairBehaviorSignals | None = None
    error: str | None = None


class NetworkSenderSignals(BaseModel):
    outgoing_transaction_count: int
    unique_recipient_count: int
    total_outgoing_amount: FiniteFloat


class NetworkRecipientSignals(BaseModel):
    incoming_transaction_count: int
    unique_sender_count: int
    total_incoming_amount: FiniteFloat


class NetworkRelationshipSignals(BaseModel):
    previous_transaction_count: int
    previous_transaction_amount: FiniteFloat
    relationship_status: Literal["NEW_RELATIONSHIP", "EXISTING_RELATIONSHIP"]


class NetworkPattern(BaseModel):
    pattern: Literal["HIGH_RECIPIENT_CONNECTIVITY", "HIGH_SENDER_CONNECTIVITY"]
    detected: bool


class NetworkSignals(BaseModel):
    status: Literal["available", "unavailable"]
    source: SignalSource | None = None
    transaction_id: str | None = None
    sender: NetworkSenderSignals | None = None
    recipient: NetworkRecipientSignals | None = None
    relationship: NetworkRelationshipSignals | None = None
    patterns: list[NetworkPattern] = Field(default_factory=list)
    error: str | None = None


class CounterfactualScenario(BaseModel):
    amount: FiniteFloat
    amount_ratio: FiniteFloat
    fraud_probability: FiniteFloat | None
    probability_change: FiniteFloat | None
    valid: bool


class BestCounterfactual(BaseModel):
    amount: FiniteFloat
    amount_ratio: FiniteFloat
    fraud_probability: FiniteFloat
    probability_reduction: FiniteFloat
    description: str


class InvestigationAction(BaseModel):
    code: Literal[
        "ANALYST_REVIEW",
        "ADDITIONAL_VERIFICATION",
        "RECIPIENT_VERIFICATION",
        "ACCOUNT_VERIFICATION",
        "TRANSACTION_DELAY",
        "TRANSACTION_LIMIT_REVIEW",
    ]
    label: str
    rationale: str


class CounterfactualInterpretation(BaseModel):
    """Separates what the model said from what an investigator might do."""

    model_output: str
    investigation_recommendations: list[InvestigationAction] = Field(default_factory=list)
    disclaimer: str


class CounterfactualResponse(BaseModel):
    transaction_id: str
    original_amount: FiniteFloat
    original_fraud_probability: FiniteFloat
    baseline_fraud_probability: FiniteFloat | None = None
    baseline_source: Literal["recomputed", "stored"] = "recomputed"
    baseline_matches_stored: bool | None = None
    counterfactuals: list[CounterfactualScenario]
    best_counterfactual: BestCounterfactual | None = None
    best_scenario: BestCounterfactual | None = None
    interpretation: CounterfactualInterpretation | None = None


class InvestigatorSummary(BaseModel):
    """Deterministic text built only from stored, verified analysis fields."""

    investigator_summary: str
    key_facts: list[str] = Field(default_factory=list)
    investigation_recommendations: list[InvestigationAction] = Field(default_factory=list)
    disclaimer: str


ReviewStatus = Literal[
    "OPEN",
    "UNDER_REVIEW",
    "DISMISSED",
    "ESCALATED",
    "CONFIRMED_SUSPICIOUS",
]


class ReviewCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ReviewStatus = "OPEN"
    analyst_note: str | None = Field(default=None, max_length=2000)
    decision: str | None = Field(default=None, max_length=200)


class ReviewUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ReviewStatus | None = None
    analyst_note: str | None = Field(default=None, max_length=2000)
    decision: str | None = Field(default=None, max_length=200)


class ReviewModelAssessment(BaseModel):
    """Read-only snapshot of the model output; reviews never change it."""

    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_score: FiniteFloat
    fraud_probability: FiniteFloat
    recommended_action: str


class ReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    transaction_id: str
    status: ReviewStatus
    analyst_note: str | None = None
    decision: str | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    model_assessment: ReviewModelAssessment | None = None


class ReviewListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    returned_items: int
    items: list[ReviewResponse]