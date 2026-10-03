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
    category: Literal["MODEL_SIGNAL", "ANOMALY_SIGNAL", "MODEL_ATTRIBUTION"]
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