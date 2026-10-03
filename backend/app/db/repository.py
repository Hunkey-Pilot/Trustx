from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import asc, desc, func, select
from sqlalchemy.orm import Session

from app.db.models import TransactionAnalysis
from app.schemas.transactions import (
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
        model_version=record.model_version,
        feature_count=record.feature_count,
    )