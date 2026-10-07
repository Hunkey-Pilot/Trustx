from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class TransactionAnalysis(Base):
    __tablename__ = "transaction_analyses"
    __table_args__ = (
        UniqueConstraint("transaction_id"),
        Index("ix_transaction_analyses_name_orig_step", "nameOrig", "step"),
        Index("ix_transaction_analyses_name_dest_step", "nameDest", "step"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    step: Mapped[int] = mapped_column(Integer)
    transaction_type: Mapped[str] = mapped_column("type", String(20), index=True)
    amount: Mapped[float] = mapped_column(Float)
    name_orig: Mapped[str] = mapped_column("nameOrig", String(128))
    old_balance_org: Mapped[float] = mapped_column("oldbalanceOrg", Float)
    new_balance_orig: Mapped[float] = mapped_column("newbalanceOrig", Float)
    name_dest: Mapped[str] = mapped_column("nameDest", String(128))
    old_balance_dest: Mapped[float] = mapped_column("oldbalanceDest", Float)
    new_balance_dest: Mapped[float] = mapped_column("newbalanceDest", Float)
    historical_transactions: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)

    fraud_probability: Mapped[float] = mapped_column(Float, index=True)
    anomaly_signal: Mapped[float] = mapped_column(Float)
    risk_score: Mapped[float] = mapped_column(Float, index=True)
    risk_level: Mapped[str] = mapped_column(String(20), index=True)
    recommended_action: Mapped[str] = mapped_column(Text, index=True)

    explanation: Mapped[dict[str, Any]] = mapped_column(JSONB)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONB)
    behavioral_signals: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    network_signals: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    request_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    feature_count: Mapped[int] = mapped_column(Integer)


class CaseReview(Base):
    """Analyst review of one analyzed transaction.

    A review never changes the stored model outputs (fraud_probability,
    risk_level, recommended_action); it is a separate human record.
    """

    __tablename__ = "case_reviews"
    __table_args__ = (UniqueConstraint("transaction_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("transaction_analyses.transaction_id", ondelete="CASCADE"),
        index=True,
    )
    status: Mapped[str] = mapped_column(String(32), index=True)
    analyst_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision: Mapped[str | None] = mapped_column(String(200), nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(64), index=True)
    resource: Mapped[str] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    details: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, nullable=True)
