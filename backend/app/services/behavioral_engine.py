from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db.repository import (
    get_pair_behavior_metrics,
    get_recipient_behavior_metrics,
    get_sender_behavior_metrics,
)
from app.schemas.transactions import (
    BehavioralSignals,
    PairBehaviorSignals,
    RecipientBehaviorSignals,
    SenderBehaviorSignals,
    TransactionAnalyzeRequest,
)


class BehavioralEngine:
    """Calculate contextual behavior signals from persisted history only."""

    def calculate(
        self,
        db: Session,
        transaction: TransactionAnalyzeRequest,
        transaction_id: str | None = None,
        analysis_time: datetime | None = None,
    ) -> BehavioralSignals:
        current_time = analysis_time or datetime.now(timezone.utc)
        try:
            sender = get_sender_behavior_metrics(
                db,
                name_orig=transaction.nameOrig,
                current_step=transaction.step,
                current_time=current_time,
                transaction_id=transaction_id,
            )
            recipient = get_recipient_behavior_metrics(
                db,
                name_dest=transaction.nameDest,
                current_step=transaction.step,
                current_time=current_time,
                transaction_id=transaction_id,
            )
            pair = get_pair_behavior_metrics(
                db,
                name_orig=transaction.nameOrig,
                name_dest=transaction.nameDest,
                current_step=transaction.step,
                current_time=current_time,
                transaction_id=transaction_id,
            )
        except Exception as exc:
            return BehavioralSignals(
                status="unavailable",
                error=f"Behavioral history unavailable: {type(exc).__name__}",
            )

        return BehavioralSignals(
            status="available",
            sender=SenderBehaviorSignals(
                current_amount=transaction.amount,
                transaction_count_5m=sender["count_5m"],
                transaction_count_1h=sender["count_1h"],
                transaction_count_24h=sender["count_24h"],
                amount_sum_1h=sender["sum_1h"],
                amount_sum_24h=sender["sum_24h"],
                historical_average_amount=sender["average_amount"],
                amount_to_historical_average=(
                    transaction.amount / sender["average_amount"]
                    if sender["average_amount"] not in (None, 0)
                    else None
                ),
                historical_max_amount=sender["max_amount"],
                amount_to_historical_max=(
                    transaction.amount / sender["max_amount"]
                    if sender["max_amount"] not in (None, 0)
                    else None
                ),
            ),
            recipient=RecipientBehaviorSignals(
                transaction_count=recipient["count"],
                unique_senders=recipient["unique_senders"],
                amount_sum=recipient["amount_sum"],
                transaction_count_1h=recipient["count_1h"],
                transaction_count_24h=recipient["count_24h"],
                unique_senders_1h=recipient["unique_senders_1h"],
                unique_senders_24h=recipient["unique_senders_24h"],
            ),
            sender_recipient=PairBehaviorSignals(
                pair_transaction_count=pair["count"],
                pair_amount_sum=pair["amount_sum"],
                is_new_recipient_for_sender=pair["count"] == 0,
                minutes_since_previous_pair_transaction=(
                    (current_time - pair["last_created_at"]).total_seconds() / 60
                    if pair["last_created_at"] is not None
                    else None
                ),
            ),
        )