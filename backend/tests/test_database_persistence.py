import unittest
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from fastapi import HTTPException

from app.api.transactions import (
    analyze_transaction,
    get_transaction,
    list_transactions,
)
from app.db.connection import verify_database_connection
from app.db.models import TransactionAnalysis
from app.db.repository import (
    create_transaction_analysis,
    get_transaction_analysis,
    list_transaction_analyses,
    transaction_summary,
)
from app.db.session import SessionLocal, engine
from app.main import app, model_loader
from app.schemas.transactions import (
    EvidenceResponse,
    RiskAssessment,
    ShapExplanation,
    TransactionAnalyzeRequest,
    TransactionAnalyzeResponse,
)
from app.services.evidence_engine import EvidenceEngine
from app.services.model_service import ModelService
from app.services.risk_engine import RiskEngine
from app.services.shap_service import ShapService


class DatabasePersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if verify_database_connection()["status"] != "connected":
            raise unittest.SkipTest("PostgreSQL is unavailable")
        if not inspect(engine).has_table("transaction_analyses"):
            raise unittest.SkipTest("transaction_analyses migration is not applied")
        cls.db = SessionLocal()
        cls.created_ids = []

    @classmethod
    def tearDownClass(cls):
        if not hasattr(cls, "db"):
            return
        if cls.created_ids:
            cls.db.query(TransactionAnalysis).filter(
                TransactionAnalysis.transaction_id.in_(cls.created_ids)
            ).delete(synchronize_session=False)
            cls.db.commit()
        cls.db.close()

    def make_input(
        self,
        transaction_id: str,
        transaction_type: str = "TRANSFER",
        amount: float = 100.0,
    ) -> TransactionAnalyzeRequest:
        return TransactionAnalyzeRequest(
            step=49,
            type=transaction_type,
            amount=amount,
            nameOrig="C1",
            oldbalanceOrg=300.0,
            newbalanceOrig=200.0,
            nameDest="M1",
            oldbalanceDest=50.0,
            newbalanceDest=150.0,
            historical_transactions=[],
        )

    def make_analysis(
        self,
        transaction_id: str,
        probability: float = 0.1,
        risk_score: float = 10.0,
        risk_level: str = "LOW",
        action: str = "ALLOW / MONITOR",
    ) -> TransactionAnalyzeResponse:
        return TransactionAnalyzeResponse(
            transaction_id=transaction_id,
            fraud_probability=probability,
            anomaly_signal=-0.02,
            risk_score=risk_score,
            risk_level=risk_level,
            recommended_action=action,
            explanation=ShapExplanation(
                status="available",
                method="SHAP",
                type="model_attribution",
                output_space="raw_margin",
                base_value=0.0,
                top_features=[],
            ),
            evidence=EvidenceResponse(
                status="available",
                items=[],
                risk_assessment=RiskAssessment(
                    risk_score=10.0,
                    risk_level="LOW",
                    recommended_action="ALLOW / MONITOR",
                ),
            ),
            model_version=None,
            feature_count=28,
        )

    def persist_one(
        self,
        transaction_type: str = "TRANSFER",
        amount: float = 100.0,
        probability: float = 0.1,
        risk_score: float = 10.0,
        risk_level: str = "LOW",
        action: str = "ALLOW / MONITOR",
    ) -> TransactionAnalysis:
        transaction_id = f"dbtest_{uuid4().hex}"
        self.created_ids.append(transaction_id)
        return create_transaction_analysis(
            self.db,
            self.make_input(transaction_id, transaction_type, amount),
            self.make_analysis(transaction_id, probability, risk_score, risk_level, action),
        )

    def test_database_connection_and_migrated_table(self):
        self.assertEqual(verify_database_connection()["status"], "connected")
        self.assertTrue(inspect(engine).has_table("transaction_analyses"))

    def test_insert_and_retrieve_persisted_analysis(self):
        record = self.persist_one()
        found = get_transaction_analysis(self.db, record.transaction_id)
        self.assertIsNotNone(found)
        self.assertEqual(found.fraud_probability, 0.1)
        self.assertEqual(found.evidence, self.make_analysis(record.transaction_id).evidence.model_dump())

    def test_commit_failure_rolls_back_session(self):
        record = self.persist_one()
        duplicate = self.make_analysis(record.transaction_id)
        with self.assertRaises(IntegrityError):
            create_transaction_analysis(self.db, self.make_input(record.transaction_id), duplicate)
        self.db.rollback()
        self.assertIsNotNone(get_transaction_analysis(self.db, record.transaction_id))

    def test_list_pagination(self):
        self.persist_one()
        self.persist_one()
        self.persist_one()
        first_page = list_transaction_analyses(self.db, page=1, page_size=2)
        second_page = list_transaction_analyses(self.db, page=2, page_size=2)
        self.assertGreaterEqual(first_page.total, 3)
        self.assertLessEqual(len(first_page.items), 2)
        self.assertLessEqual(len(second_page.items), 2)

    def test_analysis_route_persists_with_real_session(self):
        loaded_models = model_loader.loaded or model_loader.load()
        app.state.model_loader = model_loader
        app.state.model_service = ModelService(loaded_models)
        app.state.risk_engine = RiskEngine()
        app.state.shap_service = ShapService(loaded_models)
        app.state.evidence_engine = EvidenceEngine()
        request = SimpleNamespace(app=app)
        transaction = self.make_input(f"dbtest_{uuid4().hex}")
        response = analyze_transaction(request, transaction, db=self.db)
        self.created_ids.append(response.transaction_id)
        self.assertIsNotNone(get_transaction_analysis(self.db, response.transaction_id))

    def test_retrieval_routes_and_missing_transaction(self):
        record = self.persist_one()
        response = get_transaction(record.transaction_id, db=self.db)
        self.assertEqual(response.transaction_id, record.transaction_id)
        listed = list_transactions(db=self.db, page=1, page_size=10)
        self.assertTrue(any(item.transaction_id == record.transaction_id for item in listed.items))
        with self.assertRaises(HTTPException) as error:
            get_transaction("does-not-exist", db=self.db)
        self.assertEqual(error.exception.status_code, 404)

    def test_filters_sorting_and_summary(self):
        high = self.persist_one(
            transaction_type="CASH_OUT",
            amount=900.0,
            probability=0.9,
            risk_score=90.0,
            risk_level="CRITICAL",
            action="TEMPORARY_HOLD / ESCALATE_FOR_REVIEW",
        )
        low = self.persist_one(
            transaction_type="TRANSFER",
            amount=100.0,
            probability=0.1,
            risk_score=10.0,
        )
        filtered = list_transaction_analyses(
            self.db,
            page=1,
            page_size=10,
            risk_level="CRITICAL",
            transaction_type="CASH_OUT",
            min_fraud_probability=0.8,
            min_risk_score=80.0,
            sort_by="risk_score",
            sort_order="desc",
        )
        self.assertEqual(filtered.returned_items, 1)
        self.assertEqual(filtered.items[0].transaction_id, high.transaction_id)

        date_filtered = list_transaction_analyses(
            self.db,
            page=1,
            page_size=10,
            created_after=low.created_at - timedelta(seconds=1),
            transaction_id=low.transaction_id,
        )
        self.assertEqual(date_filtered.returned_items, 1)
        summary = transaction_summary(self.db)
        self.assertGreaterEqual(summary.total_analyzed_transactions, 2)
        self.assertGreaterEqual(summary.count_by_risk_level.get("CRITICAL", 0), 1)
        self.assertGreaterEqual(summary.count_by_transaction_type.get("CASH_OUT", 0), 1)

    def test_invalid_filter_range_returns_422(self):
        with self.assertRaises(HTTPException) as error:
            list_transactions(
                db=self.db,
                page=1,
                page_size=10,
                min_fraud_probability=0.8,
                max_fraud_probability=0.2,
            )
        self.assertEqual(error.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()