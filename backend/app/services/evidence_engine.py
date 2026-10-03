from __future__ import annotations

from dataclasses import dataclass

from app.services.risk_engine import RiskResult
from app.schemas.transactions import BehavioralSignals, NetworkSignals
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
    risk_assessment: RiskResult | None


class EvidenceEngine:
    """Aggregate verified model signals and model attributions.

    Evidence describes model observations and application-level assessment. It
    is not proof of fraud or a confirmation that fraud occurred.
    """

    def aggregate(
        self,
        fraud_probability: float,
        anomaly_signal: float,
        shap_explanation: ShapExplanation | None,
        risk_result: RiskResult | None,
        behavioral_signals: BehavioralSignals | None = None,
        network_signals: NetworkSignals | None = None,
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

        if shap_explanation is not None and shap_explanation.status == "available":
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

        if behavioral_signals is not None and behavioral_signals.status == "available":
            sender = behavioral_signals.sender
            recipient = behavioral_signals.recipient
            pair = behavioral_signals.sender_recipient
            if sender is not None:
                items.extend(
                    [
                        EvidenceItem(
                            category="BEHAVIORAL_SIGNAL",
                            type="sender_velocity",
                            value=sender.transaction_count_5m,
                            description="Persisted sender transaction count in the last 5 minutes",
                        ),
                        EvidenceItem(
                            category="BEHAVIORAL_SIGNAL",
                            type="sender_velocity",
                            value=sender.transaction_count_1h,
                            description="Persisted sender transaction count in the last hour",
                        ),
                    ]
                )
            if recipient is not None:
                items.append(
                    EvidenceItem(
                        category="BEHAVIORAL_SIGNAL",
                        type="recipient_network_activity",
                        value=recipient.unique_senders_24h,
                        description="Distinct persisted senders for the recipient in the last 24 hours",
                    )
                )
            if pair is not None:
                items.append(
                    EvidenceItem(
                        category="BEHAVIORAL_SIGNAL",
                        type="sender_recipient_relationship",
                        value=pair.pair_transaction_count,
                        description="Previous persisted transactions between this sender and recipient",
                    )
                )

        if network_signals is not None and network_signals.status == "available":
            for pattern in network_signals.patterns:
                if not pattern.detected:
                    continue
                if pattern.pattern == "HIGH_RECIPIENT_CONNECTIVITY":
                    description = (
                        "Network-based signal: High recipient connectivity observed. "
                        "Recipient has transactions from multiple unique senders."
                    )
                else:
                    description = (
                        "Network-based signal: High sender connectivity observed. "
                        "Sender has transactions with multiple unique recipients."
                    )
                items.append(
                    EvidenceItem(
                        category="NETWORK_SIGNAL",
                        type=pattern.pattern.lower(),
                        value=None,
                        description=description,
                    )
                )

        return EvidenceResult(
            status="available" if shap_explanation is not None and shap_explanation.status == "available" else "partial",
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