from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


TransactionType = Literal[
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


class HistoricalTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    amount: FiniteFloat = Field(ge=0)
    nameOrig: str = Field(min_length=1)


class TransactionAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    type: TransactionType
    amount: FiniteFloat = Field(ge=0)
    nameOrig: str = Field(min_length=1)
    oldbalanceOrg: FiniteFloat = Field(ge=0)
    newbalanceOrig: FiniteFloat = Field(ge=0)
    nameDest: str = Field(min_length=1)
    oldbalanceDest: FiniteFloat = Field(ge=0)
    newbalanceDest: FiniteFloat = Field(ge=0)
    historical_transactions: list[HistoricalTransaction] = Field(default_factory=list)


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
    historical_transactions: list[HistoricalTransaction]
    fraud_probability: FiniteFloat
    anomaly_signal: FiniteFloat
    risk_score: FiniteFloat = Field(ge=0, le=100)
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    recommended_action: str
    explanation: ShapExplanation
    evidence: EvidenceResponse
    behavioral_signals: BehavioralSignals
    network_signals: NetworkSignals | None = None
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


class BehavioralSignals(BaseModel):
    status: Literal["available", "unavailable"]
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


class CounterfactualResponse(BaseModel):
    transaction_id: str
    original_amount: FiniteFloat
    original_fraud_probability: FiniteFloat
    counterfactuals: list[CounterfactualScenario]
    best_counterfactual: BestCounterfactual | None = None