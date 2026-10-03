from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RiskEngineError(ValueError):
    """Raised when risk-engine inputs are invalid."""


@dataclass(frozen=True)
class RiskResult:
    risk_score: float
    risk_level: str
    recommended_action: str


class RiskEngine:
    """Convert model outputs into an application-level decision-support result.

    ``fraud_probability`` is the supervised model confidence signal and is
    already bounded in [0, 1]. ``anomaly_signal`` is the raw unsupervised
    Isolation Forest decision value. No production-safe calibration artifact
    exists for that raw value, so it is validated and preserved by the API but
    does not contribute to this initial risk score.
    """

    def calculate(self, fraud_probability: float, anomaly_signal: float) -> RiskResult:
        self._validate_inputs(fraud_probability, anomaly_signal)

        risk_score = round(fraud_probability * 100, 6)
        risk_level = self._risk_level(risk_score)
        return RiskResult(
            risk_score=risk_score,
            risk_level=risk_level,
            recommended_action=self._recommended_action(risk_level),
        )

    @staticmethod
    def _validate_inputs(fraud_probability: float, anomaly_signal: float) -> None:
        if not np.isfinite(fraud_probability) or not 0 <= fraud_probability <= 1:
            raise RiskEngineError("fraud_probability must be finite and within [0, 1]")
        if not np.isfinite(anomaly_signal):
            raise RiskEngineError("anomaly_signal must be finite")

    @staticmethod
    def _risk_level(risk_score: float) -> str:
        if risk_score < 30:
            return "LOW"
        if risk_score < 60:
            return "MEDIUM"
        if risk_score < 85:
            return "HIGH"
        return "CRITICAL"

    @staticmethod
    def _recommended_action(risk_level: str) -> str:
        actions = {
            "LOW": "ALLOW / MONITOR",
            "MEDIUM": "ADDITIONAL_VERIFICATION / MONITOR",
            "HIGH": "STEP_UP_VERIFICATION / MANUAL_REVIEW",
            "CRITICAL": "TEMPORARY_HOLD / ESCALATE_FOR_REVIEW",
        }
        return actions[risk_level]