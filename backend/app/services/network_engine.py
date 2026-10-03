from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.repository import get_network_metrics
from app.schemas.transactions import (
    NetworkPattern,
    NetworkSignals,
    NetworkRelationshipSignals,
    NetworkRecipientSignals,
    NetworkSenderSignals,
    TransactionAnalyzeRequest,
)


NETWORK_UNIQUE_SENDERS_THRESHOLD = int(os.getenv("NETWORK_UNIQUE_SENDERS_THRESHOLD", "5"))
NETWORK_UNIQUE_RECIPIENTS_THRESHOLD = int(os.getenv("NETWORK_UNIQUE_RECIPIENTS_THRESHOLD", "5"))


class NetworkEngine:
    """Calculate descriptive connectivity signals from prior history and this transaction."""

    UNIQUE_SENDER_THRESHOLD = NETWORK_UNIQUE_SENDERS_THRESHOLD
    UNIQUE_RECIPIENT_THRESHOLD = NETWORK_UNIQUE_RECIPIENTS_THRESHOLD

    def calculate(
        self,
        db: Session,
        transaction: TransactionAnalyzeRequest,
        transaction_id: str | None = None,
        analysis_time: datetime | None = None,
    ) -> NetworkSignals:
        current_time = analysis_time or datetime.now(timezone.utc)
        try:
            metrics = get_network_metrics(
                db,
                name_orig=transaction.nameOrig,
                name_dest=transaction.nameDest,
                amount=transaction.amount,
                current_step=transaction.step,
                current_time=current_time,
                transaction_id=transaction_id,
            )
        except Exception as exc:
            return NetworkSignals(
                status="unavailable",
                transaction_id=transaction_id,
                error=f"Network history unavailable: {type(exc).__name__}",
            )

        patterns: list[NetworkPattern] = []
        if metrics["unique_sender_count"] >= self.UNIQUE_SENDER_THRESHOLD:
            patterns.append(
                NetworkPattern(pattern="HIGH_RECIPIENT_CONNECTIVITY", detected=True)
            )
        if metrics["unique_recipient_count"] >= self.UNIQUE_RECIPIENT_THRESHOLD:
            patterns.append(
                NetworkPattern(pattern="HIGH_SENDER_CONNECTIVITY", detected=True)
            )

        return NetworkSignals(
            status="available",
            transaction_id=transaction_id,
            sender=NetworkSenderSignals(
                outgoing_transaction_count=metrics["outgoing_transaction_count"],
                unique_recipient_count=metrics["unique_recipient_count"],
                total_outgoing_amount=metrics["total_outgoing_amount"],
            ),
            recipient=NetworkRecipientSignals(
                incoming_transaction_count=metrics["incoming_transaction_count"],
                unique_sender_count=metrics["unique_sender_count"],
                total_incoming_amount=metrics["total_incoming_amount"],
            ),
            relationship=NetworkRelationshipSignals(
                previous_transaction_count=metrics["previous_transaction_count"],
                previous_transaction_amount=metrics["previous_transaction_amount"],
                relationship_status=(
                    "EXISTING_RELATIONSHIP"
                    if metrics["previous_transaction_count"] > 0
                    else "NEW_RELATIONSHIP"
                ),
            ),
            patterns=patterns,
        )
