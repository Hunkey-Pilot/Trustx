from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from app.schemas.transactions import (
    BestCounterfactual,
    CounterfactualInterpretation,
    CounterfactualResponse,
    CounterfactualScenario,
    TransactionAnalyzeRequest,
)
from app.services.feature_engineering import create_model_features
from app.services.model_service import ModelService
from app.services.recommendations import RECOMMENDATION_DISCLAIMER, build_recommendations


COUNTERFACTUAL_MIN_PROBABILITY_REDUCTION = float(
    os.getenv("COUNTERFACTUAL_MIN_PROBABILITY_REDUCTION", "0.05")
)
AMOUNT_RATIOS = (1.0, 0.75, 0.50, 0.25)
BASELINE_MATCH_TOLERANCE = 1e-3

COUNTERFACTUAL_DISCLAIMER = (
    "Counterfactual scenarios are hypothetical model responses, not causal "
    "explanations. A lower amount does not make a transaction legitimate and "
    "does not guarantee that fraud would be prevented."
)


class CounterfactualEngine:
    """Run deterministic amount-only hypothetical predictions on copied inputs.

    Balance semantics per transaction type (derived from the PaySim data; the
    fraction of rows following each rule is documented in the README):

    * TRANSFER, CASH_OUT, DEBIT: sender balance falls by the amount and the
      recipient balance rises by the amount.
    * PAYMENT: sender balance falls by the amount; the recipient is a merchant
      whose balance PaySim does not track, so it is left unchanged.
    * CASH_IN: sender balance RISES by the amount and the counter-party
      balance falls by the amount.

    Every scenario (including the 100% baseline) is recomputed with the same
    rules and fed through the exact 28-feature pipeline and the saved XGBoost
    model. Because recomputation makes both balance-error features zero, the
    recomputed baseline can differ from the stored probability when the
    original balances were not internally consistent. Scenario changes are
    therefore measured against the recomputed baseline.
    """

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
        risk_level: str | None = None,
        behavioral_signals: Mapping[str, Any] | None = None,
        network_signals: Mapping[str, Any] | None = None,
    ) -> CounterfactualResponse:
        original_values = transaction.model_dump(exclude={"historical_transactions"})
        historical_context = pd.DataFrame(
            [item.model_dump() for item in transaction.historical_transactions],
            columns=["step", "amount", "nameOrig"],
        )

        # Pass 1: score every scenario with the shared recompute rules.
        scored: list[tuple[float, float, float | None]] = []
        for ratio in AMOUNT_RATIOS:
            candidate_amount = transaction.amount * ratio
            candidate = self._hypothetical_transaction(original_values, candidate_amount)
            if candidate is None:
                scored.append((ratio, candidate_amount, None))
                continue
            features = create_model_features(
                transaction=candidate,
                historical_context=historical_context,
                feature_list=feature_list,
            )
            scored.append(
                (ratio, candidate_amount, model_service.predict(features).fraud_probability)
            )

        baseline_probability = next(
            (probability for ratio, _, probability in scored if ratio == 1.0), None
        )
        baseline_source = "recomputed"
        reference = baseline_probability
        if reference is None:
            baseline_source = "stored"
            reference = current_probability
        baseline_matches_stored = (
            abs(baseline_probability - current_probability) <= BASELINE_MATCH_TOLERANCE
            if baseline_probability is not None
            else None
        )

        # Pass 2: compare against the baseline.
        scenarios: list[CounterfactualScenario] = []
        best: tuple[float, float, float, float] | None = None
        for ratio, candidate_amount, probability in scored:
            if probability is None:
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
            scenarios.append(
                CounterfactualScenario(
                    amount=candidate_amount,
                    amount_ratio=ratio,
                    fraud_probability=probability,
                    probability_change=probability - reference,
                    valid=True,
                )
            )
            reduction = reference - probability
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

        interpretation = self._interpret(
            best_counterfactual=best_counterfactual,
            reference=reference,
            baseline_source=baseline_source,
            risk_level=risk_level,
            behavioral_signals=behavioral_signals,
            network_signals=network_signals,
        )
        return CounterfactualResponse(
            transaction_id=transaction_id,
            original_amount=transaction.amount,
            original_fraud_probability=current_probability,
            baseline_fraud_probability=baseline_probability,
            baseline_source=baseline_source,
            baseline_matches_stored=baseline_matches_stored,
            counterfactuals=scenarios,
            best_counterfactual=best_counterfactual,
            best_scenario=best_counterfactual,
            interpretation=interpretation,
        )

    def _interpret(
        self,
        best_counterfactual: BestCounterfactual | None,
        reference: float,
        baseline_source: str,
        risk_level: str | None,
        behavioral_signals: Mapping[str, Any] | None,
        network_signals: Mapping[str, Any] | None,
    ) -> CounterfactualInterpretation:
        basis = (
            "the recomputed 100% baseline"
            if baseline_source == "recomputed"
            else "the stored model score (the 100% scenario was not valid)"
        )
        if best_counterfactual is not None:
            model_output = (
                f"MODEL OUTPUT: at {best_counterfactual.amount_ratio * 100:.0f}% of the amount the "
                f"model score would be {best_counterfactual.fraud_probability:.4f} instead of "
                f"{reference:.4f} ({basis}), a reduction of "
                f"{best_counterfactual.probability_reduction:.4f}. This only describes how the "
                "model responds to a changed input."
            )
        else:
            model_output = (
                "MODEL OUTPUT: no smaller-amount scenario reduced the model score by at least "
                f"{self.min_probability_reduction:.2f} relative to {basis}."
            )
        recommendations = (
            build_recommendations(
                risk_level,
                behavioral_signals=behavioral_signals,
                network_signals=network_signals,
                amount_sensitive=best_counterfactual is not None,
            )
            if risk_level
            else []
        )
        return CounterfactualInterpretation(
            model_output=model_output,
            investigation_recommendations=recommendations,
            disclaimer=f"{COUNTERFACTUAL_DISCLAIMER} {RECOMMENDATION_DISCLAIMER}",
        )

    @staticmethod
    def _hypothetical_transaction(
        original: dict[str, Any],
        amount: float,
    ) -> dict[str, Any] | None:
        old_sender = original["oldbalanceOrg"]
        old_recipient = original["oldbalanceDest"]
        transaction_type = original["type"]

        if transaction_type == "CASH_IN":
            new_sender = old_sender + amount
            new_recipient = old_recipient - amount
        elif transaction_type == "PAYMENT":
            new_sender = old_sender - amount
            new_recipient = old_recipient
        else:  # TRANSFER, CASH_OUT, DEBIT
            new_sender = old_sender - amount
            new_recipient = old_recipient + amount

        if (
            not all(math.isfinite(value) for value in (amount, new_sender, new_recipient))
            or new_sender < 0
            or new_recipient < 0
        ):
            return None

        candidate = dict(original)
        candidate["amount"] = amount
        candidate["newbalanceOrig"] = new_sender
        candidate["newbalanceDest"] = new_recipient
        return candidate
