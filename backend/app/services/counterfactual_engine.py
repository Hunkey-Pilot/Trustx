from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Any

import pandas as pd

from app.schemas.transactions import (
    BestCounterfactual,
    CounterfactualResponse,
    CounterfactualScenario,
    TransactionAnalyzeRequest,
)
from app.services.feature_engineering import create_model_features
from app.services.model_service import ModelService


COUNTERFACTUAL_MIN_PROBABILITY_REDUCTION = float(
    os.getenv("COUNTERFACTUAL_MIN_PROBABILITY_REDUCTION", "0.05")
)
AMOUNT_RATIOS = (1.0, 0.75, 0.50, 0.25)


class CounterfactualEngine:
    """Run deterministic amount-only hypothetical predictions on copied inputs."""

    def __init__(
        self,
        min_probability_reduction: float = COUNTERFACTUAL_MIN_PROBABILITY_REDUCTION,
    ) -> None:
        if (
            not math.isfinite(min_probability_reduction)
            or min_probability_reduction < 0
            or min_probability_reduction > 1
        ):
            raise ValueError("min_probability_reduction must be between 0 and 1")
        self.min_probability_reduction = min_probability_reduction

    def evaluate(
        self,
        transaction_id: str,
        transaction: TransactionAnalyzeRequest,
        current_probability: float,
        feature_list: Sequence[str],
        model_service: ModelService,
    ) -> CounterfactualResponse:
        original_values = transaction.model_dump(exclude={"historical_transactions"})
        historical_context = pd.DataFrame(
            [item.model_dump() for item in transaction.historical_transactions],
            columns=["step", "amount", "nameOrig"],
        )
        scenarios: list[CounterfactualScenario] = []
        best: tuple[float, float, float, float] | None = None

        for ratio in AMOUNT_RATIOS:
            candidate_amount = transaction.amount * ratio
            candidate = self._hypothetical_transaction(original_values, candidate_amount)
            if candidate is None:
                scenarios.append(
                    CounterfactualScenario(
                        amount=candidate_amount,
                        amount_ratio=ratio,
                        fraud_probability=None,
                        probability_change=None,
                        valid=False,
                    )
                )
                continue

            features = create_model_features(
                transaction=candidate,
                historical_context=historical_context,
                feature_list=feature_list,
            )
            probability = model_service.predict(features).fraud_probability
            probability_change = probability - current_probability
            scenarios.append(
                CounterfactualScenario(
                    amount=candidate_amount,
                    amount_ratio=ratio,
                    fraud_probability=probability,
                    probability_change=probability_change,
                    valid=True,
                )
            )

            reduction = current_probability - probability
            if ratio < 1.0 and (best is None or reduction > best[3]):
                best = (candidate_amount, ratio, probability, reduction)

        best_counterfactual = None
        if best is not None and best[3] >= self.min_probability_reduction:
            best_counterfactual = BestCounterfactual(
                amount=best[0],
                amount_ratio=best[1],
                fraud_probability=best[2],
                probability_reduction=best[3],
                description=(
                    "Hypothetical scenario with lower model-predicted fraud probability."
                ),
            )

        return CounterfactualResponse(
            transaction_id=transaction_id,
            original_amount=transaction.amount,
            original_fraud_probability=current_probability,
            counterfactuals=scenarios,
            best_counterfactual=best_counterfactual,
        )

    @staticmethod
    def _hypothetical_transaction(
        original: dict[str, Any],
        amount: float,
    ) -> dict[str, Any] | None:
        candidate = dict(original)
        new_sender_balance = candidate["oldbalanceOrg"] - amount
        new_recipient_balance = candidate["oldbalanceDest"] + amount
        if (
            not all(math.isfinite(value) for value in (amount, new_sender_balance, new_recipient_balance))
            or new_sender_balance < 0
            or new_recipient_balance < 0
        ):
            return None

        candidate["amount"] = amount
        candidate["newbalanceOrig"] = new_sender_balance
        candidate["newbalanceDest"] = new_recipient_balance
        return candidate
