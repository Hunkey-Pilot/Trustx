from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import asc, desc, func, select
from sqlalchemy.orm import Session

from app.db.models import TransactionAnalysis
from app.schemas.transactions import (
    BehavioralSignals,
    PersistedTransactionResponse,
    TransactionAnalyzeRequest,
    TransactionAnalyzeResponse,
    TransactionListResponse,
    TransactionSummaryResponse,
)


def create_transaction_analysis(
    db: Session,
    transaction: TransactionAnalyzeRequest,
    analysis: TransactionAnalyzeResponse,
) -> TransactionAnalysis:
    input_data = transaction.model_dump(mode="json")
    output_data = analysis.model_dump(mode="json")
    record = TransactionAnalysis(
        transaction_id=analysis.transaction_id,
        step=input_data["step"],
        transaction_type=input_data["type"],
        amount=input_data["amount"],
        name_orig=input_data["nameOrig"],
        old_balance_org=input_data["oldbalanceOrg"],
        new_balance_orig=input_data["newbalanceOrig"],
        name_dest=input_data["nameDest"],
        old_balance_dest=input_data["oldbalanceDest"],
        new_balance_dest=input_data["newbalanceDest"],
        historical_transactions=input_data["historical_transactions"],
        fraud_probability=output_data["fraud_probability"],
        anomaly_signal=output_data["anomaly_signal"],
        risk_score=output_data["risk_score"],
        risk_level=output_data["risk_level"],
        recommended_action=output_data["recommended_action"],
        explanation=output_data["explanation"],
        evidence=output_data["evidence"],
        behavioral_signals=output_data.get("behavioral_signals"),
        model_version=output_data["model_version"],
        feature_count=output_data["feature_count"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_transaction_analysis(
    db: Session,
    transaction_id: str,
) -> TransactionAnalysis | None:
    return db.scalar(
        select(TransactionAnalysis).where(
            TransactionAnalysis.transaction_id == transaction_id
        )
    )


def list_transaction_analyses(
    db: Session,
    page: int,
    page_size: int,
    risk_level: str | None = None,
    recommended_action: str | None = None,
    transaction_type: str | None = None,
    transaction_id: str | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    min_fraud_probability: float | None = None,
    max_fraud_probability: float | None = None,
    min_risk_score: float | None = None,
    max_risk_score: float | None = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
) -> TransactionListResponse:
    filters = _transaction_filters(
        risk_level=risk_level,
        recommended_action=recommended_action,
        transaction_type=transaction_type,
        transaction_id=transaction_id,
        created_after=created_after,
        created_before=created_before,
        min_fraud_probability=min_fraud_probability,
        max_fraud_probability=max_fraud_probability,
        min_risk_score=min_risk_score,
        max_risk_score=max_risk_score,
    )
    base_query = select(TransactionAnalysis).where(*filters)
    total = db.scalar(
        select(func.count()).select_from(base_query.subquery())
    ) or 0
    sort_column = {
        "created_at": TransactionAnalysis.created_at,
        "fraud_probability": TransactionAnalysis.fraud_probability,
        "risk_score": TransactionAnalysis.risk_score,
        "amount": TransactionAnalysis.amount,
    }[sort_by]
    ordering = desc(sort_column) if sort_order == "desc" else asc(sort_column)
    records = db.scalars(
        base_query.order_by(ordering, TransactionAnalysis.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return TransactionListResponse(
        total=total,
        page=page,
        page_size=page_size,
        returned_items=len(records),
        items=[transaction_analysis_to_response(record) for record in records],
    )


def transaction_summary(db: Session) -> TransactionSummaryResponse:
    total = db.scalar(select(func.count()).select_from(TransactionAnalysis)) or 0
    risk_counts = db.execute(
        select(TransactionAnalysis.risk_level, func.count())
        .group_by(TransactionAnalysis.risk_level)
    ).all()
    action_counts = db.execute(
        select(TransactionAnalysis.recommended_action, func.count())
        .group_by(TransactionAnalysis.recommended_action)
    ).all()
    type_counts = db.execute(
        select(TransactionAnalysis.transaction_type, func.count())
        .group_by(TransactionAnalysis.transaction_type)
    ).all()
    aggregates = db.execute(
        select(
            func.avg(TransactionAnalysis.fraud_probability),
            func.max(TransactionAnalysis.fraud_probability),
            func.avg(TransactionAnalysis.risk_score),
            func.max(TransactionAnalysis.risk_score),
        )
    ).one()
    return TransactionSummaryResponse(
        total_analyzed_transactions=total,
        count_by_risk_level={key: count for key, count in risk_counts},
        count_by_recommended_action={key: count for key, count in action_counts},
        count_by_transaction_type={key: count for key, count in type_counts},
        average_fraud_probability=aggregates[0],
        maximum_fraud_probability=aggregates[1],
        average_risk_score=aggregates[2],
        maximum_risk_score=aggregates[3],
    )


def _transaction_filters(**kwargs: Any) -> list[Any]:
    filters: list[Any] = []
    mapping = {
        "risk_level": TransactionAnalysis.risk_level,
        "recommended_action": TransactionAnalysis.recommended_action,
        "transaction_type": TransactionAnalysis.transaction_type,
        "transaction_id": TransactionAnalysis.transaction_id,
    }
    for key, column in mapping.items():
        if kwargs[key] is not None:
            filters.append(column == kwargs[key])
    range_mapping = {
        "created_after": (TransactionAnalysis.created_at, ">"),
        "created_before": (TransactionAnalysis.created_at, "<"),
        "min_fraud_probability": (TransactionAnalysis.fraud_probability, ">="),
        "max_fraud_probability": (TransactionAnalysis.fraud_probability, "<="),
        "min_risk_score": (TransactionAnalysis.risk_score, ">="),
        "max_risk_score": (TransactionAnalysis.risk_score, "<="),
    }
    for key, (column, operator) in range_mapping.items():
        value = kwargs[key]
        if value is not None:
            filters.append(getattr(column, {">": "__gt__", "<": "__lt__", ">=": "__ge__", "<=": "__le__"}[operator])(value))
    return filters


def _history_clause(
    current_step: int,
    current_time: datetime,
    transaction_id: str | None,
) -> list[Any]:
    conditions = [
        (TransactionAnalysis.step < current_step)
        | (
            (TransactionAnalysis.step == current_step)
            & (TransactionAnalysis.created_at < current_time)
        )
    ]
    if transaction_id is not None:
        conditions.append(TransactionAnalysis.transaction_id != transaction_id)
    return conditions


def _window_aggregate(
    db: Session,
    filters: list[Any],
    start_time: datetime | None = None,
) -> tuple[int, float]:
    window_filters = list(filters)
    if start_time is not None:
        window_filters.append(TransactionAnalysis.created_at >= start_time)
    result = db.execute(
        select(
            func.count(TransactionAnalysis.id),
            func.coalesce(func.sum(TransactionAnalysis.amount), 0.0),
        ).where(*window_filters)
    ).one()
    return int(result[0]), float(result[1] or 0.0)


def get_sender_behavior_metrics(
    db: Session,
    name_orig: str,
    current_step: int,
    current_time: datetime,
    transaction_id: str | None = None,
) -> dict[str, Any]:
    filters = [TransactionAnalysis.name_orig == name_orig] + _history_clause(
        current_step, current_time, transaction_id
    )
    count_5m, sum_5m = _window_aggregate(db, filters, current_time - timedelta(minutes=5))
    count_1h, sum_1h = _window_aggregate(db, filters, current_time - timedelta(hours=1))
    count_24h, sum_24h = _window_aggregate(db, filters, current_time - timedelta(days=1))
    historical = db.execute(
        select(
            func.avg(TransactionAnalysis.amount),
            func.max(TransactionAnalysis.amount),
        ).where(*filters)
    ).one()
    return {
        "count_5m": count_5m,
        "count_1h": count_1h,
        "count_24h": count_24h,
        "sum_5m": sum_5m,
        "sum_1h": sum_1h,
        "sum_24h": sum_24h,
        "average_amount": float(historical[0]) if historical[0] is not None else None,
        "max_amount": float(historical[1]) if historical[1] is not None else None,
    }


def get_recipient_behavior_metrics(
    db: Session,
    name_dest: str,
    current_step: int,
    current_time: datetime,
    transaction_id: str | None = None,
) -> dict[str, Any]:
    filters = [TransactionAnalysis.name_dest == name_dest] + _history_clause(
        current_step, current_time, transaction_id
    )
    total = db.execute(
        select(
            func.count(TransactionAnalysis.id),
            func.count(func.distinct(TransactionAnalysis.name_orig)),
            func.coalesce(func.sum(TransactionAnalysis.amount), 0.0),
        ).where(*filters)
    ).one()
    count_1h, _ = _window_aggregate(db, filters, current_time - timedelta(hours=1))
    count_24h, _ = _window_aggregate(db, filters, current_time - timedelta(days=1))
    unique_1h = db.scalar(
        select(func.count(func.distinct(TransactionAnalysis.name_orig))).where(
            *filters, TransactionAnalysis.created_at >= current_time - timedelta(hours=1)
        )
    ) or 0
    unique_24h = db.scalar(
        select(func.count(func.distinct(TransactionAnalysis.name_orig))).where(
            *filters, TransactionAnalysis.created_at >= current_time - timedelta(days=1)
        )
    ) or 0
    return {
        "count": int(total[0]),
        "unique_senders": int(total[1]),
        "amount_sum": float(total[2] or 0.0),
        "count_1h": count_1h,
        "count_24h": count_24h,
        "unique_senders_1h": int(unique_1h),
        "unique_senders_24h": int(unique_24h),
    }


def get_pair_behavior_metrics(
    db: Session,
    name_orig: str,
    name_dest: str,
    current_step: int,
    current_time: datetime,
    transaction_id: str | None = None,
) -> dict[str, Any]:
    filters = [
        TransactionAnalysis.name_orig == name_orig,
        TransactionAnalysis.name_dest == name_dest,
    ] + _history_clause(current_step, current_time, transaction_id)
    result = db.execute(
        select(
            func.count(TransactionAnalysis.id),
            func.coalesce(func.sum(TransactionAnalysis.amount), 0.0),
            func.max(TransactionAnalysis.created_at),
        ).where(*filters)
    ).one()
    return {
        "count": int(result[0]),
        "amount_sum": float(result[1] or 0.0),
        "last_created_at": result[2],
    }


def get_network_metrics(
    db: Session,
    name_orig: str,
    name_dest: str,
    amount: float,
    current_step: int,
    current_time: datetime,
    transaction_id: str | None = None,
) -> dict[str, Any]:
    history = _history_clause(current_step, current_time, transaction_id)
    sender_history = db.execute(
        select(
            func.count(TransactionAnalysis.id),
            func.count(func.distinct(TransactionAnalysis.name_dest)),
            func.coalesce(func.sum(TransactionAnalysis.amount), 0.0),
        ).where(TransactionAnalysis.name_orig == name_orig, *history)
    ).one()
    recipient_history = db.execute(
        select(
            func.count(TransactionAnalysis.id),
            func.count(func.distinct(TransactionAnalysis.name_orig)),
            func.coalesce(func.sum(TransactionAnalysis.amount), 0.0),
        ).where(TransactionAnalysis.name_dest == name_dest, *history)
    ).one()
    pair_history = db.execute(
        select(
            func.count(TransactionAnalysis.id),
            func.coalesce(func.sum(TransactionAnalysis.amount), 0.0),
        ).where(
            TransactionAnalysis.name_orig == name_orig,
            TransactionAnalysis.name_dest == name_dest,
            *history,
        )
    ).one()

    has_previous_relationship = int(pair_history[0]) > 0
    return {
        "outgoing_transaction_count": int(sender_history[0]) + 1,
        "unique_recipient_count": int(sender_history[1]) + int(not has_previous_relationship),
        "total_outgoing_amount": float(sender_history[2] or 0.0) + amount,
        "incoming_transaction_count": int(recipient_history[0]) + 1,
        "unique_sender_count": int(recipient_history[1]) + int(not has_previous_relationship),
        "total_incoming_amount": float(recipient_history[2] or 0.0) + amount,
        "previous_transaction_count": int(pair_history[0]),
        "previous_transaction_amount": float(pair_history[1] or 0.0),
    }


def transaction_analysis_to_response(
    record: TransactionAnalysis,
) -> PersistedTransactionResponse:
    return PersistedTransactionResponse(
        id=record.id,
        created_at=record.created_at,
        transaction_id=record.transaction_id,
        step=record.step,
        type=record.transaction_type,
        amount=record.amount,
        nameOrig=record.name_orig,
        oldbalanceOrg=record.old_balance_org,
        newbalanceOrig=record.new_balance_orig,
        nameDest=record.name_dest,
        oldbalanceDest=record.old_balance_dest,
        newbalanceDest=record.new_balance_dest,
        historical_transactions=record.historical_transactions or [],
        fraud_probability=record.fraud_probability,
        anomaly_signal=record.anomaly_signal,
        risk_score=record.risk_score,
        risk_level=record.risk_level,
        recommended_action=record.recommended_action,
        explanation=record.explanation,
        evidence=record.evidence,
        behavioral_signals=record.behavioral_signals or BehavioralSignals(
            status="unavailable", error="Behavioral signals were not stored"
        ).model_dump(),
        model_version=record.model_version,
        feature_count=record.feature_count,
    )