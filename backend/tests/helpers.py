"""Shared helpers for tests that exercise the real API with a real PostgreSQL DB."""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import inspect

from app.db.connection import verify_database_connection
from app.db.models import AuditLog, CaseReview, TransactionAnalysis
from app.db.session import SessionLocal, engine
from app.main import app
from app.services.rate_limit import reset_rate_limiter


ANALYST_KEY = "test-analyst-key-not-a-real-secret"
ADMIN_KEY = "test-admin-key-not-a-real-secret"
ANALYST = {"X-API-Key": ANALYST_KEY}
ADMIN = {"X-API-Key": ADMIN_KEY}


def unique_payload(**overrides) -> dict:
    token = uuid4().hex[:12]
    payload = {
        "step": 49,
        "type": "TRANSFER",
        "amount": 100.0,
        "nameOrig": f"CT{token}",
        "oldbalanceOrg": 300.0,
        "newbalanceOrig": 200.0,
        "nameDest": f"MT{token}",
        "oldbalanceDest": 50.0,
        "newbalanceDest": 150.0,
        "historical_transactions": [],
    }
    payload.update(overrides)
    return payload


class ApiTestCase(unittest.TestCase):
    """Boots the app (models + lifespan) once per class with test API keys."""

    @classmethod
    def setUpClass(cls):
        if verify_database_connection()["status"] != "connected":
            raise unittest.SkipTest("PostgreSQL is unavailable")
        inspector = inspect(engine)
        for table in ("transaction_analyses", "case_reviews", "audit_logs"):
            if not inspector.has_table(table):
                raise unittest.SkipTest(f"{table} migration is not applied")
        cls._env = patch.dict(
            os.environ,
            {
                "TRUSTX_ANALYST_API_KEY": ANALYST_KEY,
                "TRUSTX_ADMIN_API_KEY": ADMIN_KEY,
                "TRUSTX_RATE_LIMIT_PER_MINUTE": "100000",
            },
        )
        cls._env.start()
        cls._client_cm = TestClient(app, raise_server_exceptions=False)
        cls.client = cls._client_cm.__enter__()
        cls.db = SessionLocal()
        cls.tx_ids: list[str] = []

    @classmethod
    def tearDownClass(cls):
        if not hasattr(cls, "db"):
            return
        cls.db.rollback()
        if cls.tx_ids:
            cls.db.query(AuditLog).filter(AuditLog.resource_id.in_(cls.tx_ids)).delete(
                synchronize_session=False
            )
            cls.db.query(CaseReview).filter(CaseReview.transaction_id.in_(cls.tx_ids)).delete(
                synchronize_session=False
            )
            cls.db.query(TransactionAnalysis).filter(
                TransactionAnalysis.transaction_id.in_(cls.tx_ids)
            ).delete(synchronize_session=False)
            cls.db.commit()
        cls.db.close()
        cls._client_cm.__exit__(None, None, None)
        cls._env.stop()

    def setUp(self):
        reset_rate_limiter()

    def analyze(self, payload: dict | None = None, headers: dict | None = None):
        response = self.client.post(
            "/api/v1/transactions/analyze",
            json=payload or unique_payload(),
            headers=headers or ANALYST,
        )
        if response.status_code == 200:
            self.tx_ids.append(response.json()["transaction_id"])
        return response

    def persist_row(
        self,
        sender: str,
        recipient: str,
        step: int,
        created_at: datetime | None = None,
        amount: float = 100.0,
        risk_level: str = "LOW",
        probability: float = 0.1,
        network_signals: dict | None = None,
        behavioral_signals: dict | None = None,
    ) -> TransactionAnalysis:
        transaction_id = f"tt_{uuid4().hex}"
        self.tx_ids.append(transaction_id)
        record = TransactionAnalysis(
            transaction_id=transaction_id,
            created_at=created_at or datetime.now(timezone.utc) - timedelta(minutes=1),
            step=step,
            transaction_type="TRANSFER",
            amount=amount,
            name_orig=sender,
            old_balance_org=1000.0,
            new_balance_orig=1000.0 - amount,
            name_dest=recipient,
            old_balance_dest=100.0,
            new_balance_dest=100.0 + amount,
            historical_transactions=[],
            fraud_probability=probability,
            anomaly_signal=-0.01,
            risk_score=probability * 100,
            risk_level=risk_level,
            recommended_action="ALLOW / MONITOR",
            explanation={
                "status": "unavailable",
                "method": "SHAP",
                "type": "model_attribution",
                "output_space": None,
                "base_value": None,
                "top_features": [],
                "error": None,
            },
            evidence={
                "status": "partial",
                "items": [],
                "risk_assessment": {
                    "risk_score": probability * 100,
                    "risk_level": risk_level,
                    "recommended_action": "ALLOW / MONITOR",
                },
            },
            behavioral_signals=behavioral_signals,
            network_signals=network_signals,
            model_version=None,
            feature_count=28,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record
