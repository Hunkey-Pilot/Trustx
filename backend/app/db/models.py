from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class TransactionAnalysis(Base):
    __tablename__ = "transaction_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
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
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    feature_count: Mapped[int] = mapped_column(Integer)