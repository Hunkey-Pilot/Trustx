"""Deterministic investigation recommendations.

These are suggestions for a human analyst derived from stored analysis facts.
They are NOT model output and do not claim that a transaction is fraudulent or
that any action would prevent fraud.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.schemas.transactions import InvestigationAction


SENDER_AMOUNT_MULTIPLE_NOTABLE = 3.0
SENDER_VELOCITY_1H_NOTABLE = 3

RECOMMENDATION_DISCLAIMER = (
    "Investigation recommendations are suggestions for a human analyst based on "
    "stored signals. They are separate from the model output and do not confirm "
    "fraud."
)


def _action(code: str, label: str, rationale: str) -> InvestigationAction:
    return InvestigationAction(code=code, label=label, rationale=rationale)


def _get(mapping: Any, *path: str) -> Any:
    current = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def build_recommendations(
    risk_level: str,
    behavioral_signals: Mapping[str, Any] | None = None,
    network_signals: Mapping[str, Any] | None = None,
    amount_sensitive: bool = False,
) -> list[InvestigationAction]:
    """Map risk level and verified signals to investigation actions.

    ``amount_sensitive`` is True when the counterfactual model scenarios showed
    a notable score change at smaller amounts. That is a statement about model
    behaviour only.
    """
    actions: list[InvestigationAction] = []

    if risk_level in ("HIGH", "CRITICAL"):
        actions.append(
            _action(
                "ANALYST_REVIEW",
                "Analyst review",
                f"{risk_level} risk assessments should be reviewed by a human "
                "analyst before any decision is taken.",
            )
        )
    if risk_level in ("MEDIUM", "HIGH", "CRITICAL"):
        actions.append(
            _action(
                "ADDITIONAL_VERIFICATION",
                "Additional customer verification",
                "Confirm that the sender initiated the transaction through a "
                "channel independent of the transaction itself.",
            )
        )
    if risk_level == "CRITICAL":
        actions.append(
            _action(
                "TRANSACTION_DELAY",
                "Temporary delay pending review",
                "A short delay gives the analyst time to review before funds move.",
            )
        )

    relationship = _get(network_signals, "relationship", "relationship_status")
    patterns = {
        item.get("pattern")
        for item in (_get(network_signals, "patterns") or [])
        if isinstance(item, Mapping) and item.get("detected")
    }
    if risk_level != "LOW" and (
        relationship == "NEW_RELATIONSHIP" or "HIGH_RECIPIENT_CONNECTIVITY" in patterns
    ):
        reason = (
            "No earlier analyzed transaction exists between this sender and recipient."
            if relationship == "NEW_RELATIONSHIP"
            else "The recipient has transactions from many distinct senders."
        )
        actions.append(
            _action(
                "RECIPIENT_VERIFICATION",
                "Recipient verification",
                reason,
            )
        )

    sender_velocity = _get(behavioral_signals, "sender", "transaction_count_1h") or 0
    if risk_level != "LOW" and (
        "HIGH_SENDER_CONNECTIVITY" in patterns
        or sender_velocity >= SENDER_VELOCITY_1H_NOTABLE
    ):
        actions.append(
            _action(
                "ACCOUNT_VERIFICATION",
                "Sender account verification",
                "The sender shows elevated recent activity or sends to many "
                "distinct recipients.",
            )
        )

    amount_multiple = _get(behavioral_signals, "sender", "amount_to_historical_average")
    if risk_level != "LOW" and (
        amount_sensitive
        or (
            isinstance(amount_multiple, (int, float))
            and amount_multiple >= SENDER_AMOUNT_MULTIPLE_NOTABLE
        )
    ):
        actions.append(
            _action(
                "TRANSACTION_LIMIT_REVIEW",
                "Review transaction limit",
                "The amount is unusual relative to prior activity or the model "
                "score is sensitive to the amount; consider whether a limit "
                "applies. A lower amount would not by itself make the "
                "transaction legitimate.",
            )
        )

    return actions
