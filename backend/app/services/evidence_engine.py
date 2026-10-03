from __future__ import annotations

from dataclasses import dataclass

from app.services.risk_engine import RiskResult
from app.services.shap_service import ShapExplanation


@dataclass(frozen=True)
class EvidenceItem:
    category: str
    type: str
    value: float | None
    description: str
    feature: str | None = None
    shap_value: float | None = None
    direction: str | None = None


@dataclass(frozen=True)
class EvidenceResult:
    status: str
    items: list[EvidenceItem]
    risk_assessment: RiskResult


class EvidenceEngine:
    """Aggregate verified model signals and model attributions.

    Evidence describes model observations and application-level assessment. It
    is not proof of fraud or a confirmation that fraud occurred.
    """

    def aggregate(
        self,
        fraud_probability: float,
        anomaly_signal: float,
        shap_explanation: ShapExplanation,
        risk_result: RiskResult,
    ) -> EvidenceResult:
        items = [
            EvidenceItem(
                category="MODEL_SIGNAL",
                type="fraud_probability",
                value=fraud_probability,
                description="XGBoost fraud probability model signal",
            ),
            EvidenceItem(
                category="ANOMALY_SIGNAL",
                type="isolation_forest_decision",
                value=anomaly_signal,
                description="Raw Isolation Forest anomaly decision value",
            ),
        ]

        if shap_explanation.status == "available":
            items.extend(
                EvidenceItem(
                    category="MODEL_ATTRIBUTION",
                    type="shap_feature",
                    value=feature.value,
                    shap_value=feature.shap_value,
                    direction=feature.direction,
                    feature=feature.feature,
                    description=self._shap_description(feature.direction),
                )
                for feature in shap_explanation.top_features
            )

        return EvidenceResult(
            status="available" if shap_explanation.status == "available" else "partial",
            items=items,
            risk_assessment=risk_result,
        )

    @staticmethod
    def _shap_description(direction: str) -> str:
        if direction == "increases_fraud_prediction":
            return "Feature pushed the model prediction toward the fraud class"
        if direction == "decreases_fraud_prediction":
            return "Feature pushed the model prediction away from the fraud class"
        return "Feature made a neutral contribution to the model prediction"