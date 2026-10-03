import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.api.transactions import transaction_network
from app.db.connection import verify_database_connection
from app.db.models import TransactionAnalysis
from app.db.session import SessionLocal, engine
from app.services.network_engine import NetworkEngine


class NetworkAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if verify_database_connection()["status"] != "connected":
            raise unittest.SkipTest("PostgreSQL is unavailable")
        if not inspect(engine).has_table("transaction_analyses"):
            raise unittest.SkipTest("transaction_analyses migration is not applied")

    def setUp(self):
        self.db: Session = SessionLocal()
        self.created_ids: list[str] = []
        self.now = datetime.now(timezone.utc)
        self.prefix = f"network_analysis_{uuid4().hex}"

    def tearDown(self):
        if self.created_ids:
            self.db.query(TransactionAnalysis).filter(
                TransactionAnalysis.transaction_id.in_(self.created_ids)
            ).delete(synchronize_session=False)
            self.db.commit()
        self.db.close()

    def persist(
        self,
        suffix: str,
        sender: str,
        recipient: str,
        amount: float,
        step: int,
        created_at: datetime,
    ) -> TransactionAnalysis:
        transaction_id = f"{self.prefix}_{suffix}"
        self.created_ids.append(transaction_id)
        record = TransactionAnalysis(
            transaction_id=transaction_id,
            created_at=created_at,
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
            fraud_probability=0.1,
            anomaly_signal=-0.01,
            risk_score=10.0,
            risk_level="LOW",
            recommended_action="ALLOW / MONITOR",
            explanation={},
            evidence={},
            behavioral_signals=None,
            model_version=None,
            feature_count=28,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def calculate(self, record: TransactionAnalysis):
        from app.schemas.transactions import TransactionAnalyzeRequest

        transaction = TransactionAnalyzeRequest(
            step=record.step,
            type=record.transaction_type,
            amount=record.amount,
            nameOrig=record.name_orig,
            oldbalanceOrg=record.old_balance_org,
            newbalanceOrig=record.new_balance_orig,
            nameDest=record.name_dest,
            oldbalanceDest=record.old_balance_dest,
            newbalanceDest=record.new_balance_dest,
            historical_transactions=[],
        )
        return NetworkEngine().calculate(
            self.db,
            transaction,
            transaction_id=record.transaction_id,
            analysis_time=record.created_at,
        )

    def add_previous(self, suffix: str, sender: str, recipient: str, amount: float):
        return self.persist(
            suffix,
            sender,
            recipient,
            amount,
            step=10,
            created_at=self.now - timedelta(minutes=1),
        )

    def add_current(self, sender: str, recipient: str, amount: float = 100.0):
        return self.persist(
            "current",
            sender,
            recipient,
            amount,
            step=11,
            created_at=self.now,
        )

    def test_normal_transaction_includes_current_activity(self):
        sender = f"{self.prefix}_sender"
        recipient = f"{self.prefix}_recipient"
        result = self.calculate(self.add_current(sender, recipient, 125.0))

        self.assertEqual(result.sender.outgoing_transaction_count, 1)
        self.assertEqual(result.sender.unique_recipient_count, 1)
        self.assertEqual(result.sender.total_outgoing_amount, 125.0)
        self.assertEqual(result.recipient.incoming_transaction_count, 1)
        self.assertEqual(result.recipient.unique_sender_count, 1)
        self.assertEqual(result.recipient.total_incoming_amount, 125.0)
        self.assertEqual(result.patterns, [])

    def test_existing_sender_recipient_relationship_uses_previous_history(self):
        sender = f"{self.prefix}_sender"
        recipient = f"{self.prefix}_recipient"
        self.add_previous("previous", sender, recipient, 40.0)
        result = self.calculate(self.add_current(sender, recipient, 120.0))

        self.assertEqual(result.relationship.previous_transaction_count, 1)
        self.assertEqual(result.relationship.previous_transaction_amount, 40.0)
        self.assertEqual(result.relationship.relationship_status, "EXISTING_RELATIONSHIP")
        self.assertEqual(result.sender.outgoing_transaction_count, 2)
        self.assertEqual(result.sender.total_outgoing_amount, 160.0)

    def test_new_relationship_is_reported(self):
        sender = f"{self.prefix}_sender"
        recipient = f"{self.prefix}_recipient"
        self.add_previous("different_recipient", sender, f"{recipient}_other", 30.0)
        self.add_previous("different_sender", f"{sender}_other", recipient, 20.0)
        result = self.calculate(self.add_current(sender, recipient))

        self.assertEqual(result.relationship.previous_transaction_count, 0)
        self.assertEqual(result.relationship.previous_transaction_amount, 0.0)
        self.assertEqual(result.relationship.relationship_status, "NEW_RELATIONSHIP")
        self.assertEqual(result.sender.unique_recipient_count, 2)
        self.assertEqual(result.recipient.unique_sender_count, 2)

    def test_high_recipient_connectivity_pattern(self):
        recipient = f"{self.prefix}_recipient"
        for index in range(4):
            self.add_previous(f"sender_{index}", f"{self.prefix}_sender_{index}", recipient, 10.0)
        result = self.calculate(self.add_current(f"{self.prefix}_current_sender", recipient))

        self.assertEqual(result.recipient.unique_sender_count, 5)
        self.assertEqual(
            [item.pattern for item in result.patterns],
            ["HIGH_RECIPIENT_CONNECTIVITY"],
        )
        self.assertTrue(result.patterns[0].detected)

    def test_high_sender_connectivity_pattern(self):
        sender = f"{self.prefix}_sender"
        for index in range(4):
            self.add_previous(f"recipient_{index}", sender, f"{self.prefix}_recipient_{index}", 10.0)
        result = self.calculate(self.add_current(sender, f"{self.prefix}_current_recipient"))

        self.assertEqual(result.sender.unique_recipient_count, 5)
        self.assertEqual(
            [item.pattern for item in result.patterns],
            ["HIGH_SENDER_CONNECTIVITY"],
        )
        self.assertTrue(result.patterns[0].detected)

    def test_no_pattern_below_connectivity_thresholds(self):
        sender = f"{self.prefix}_sender"
        recipient = f"{self.prefix}_recipient"
        self.add_previous("previous", sender, f"{recipient}_other", 10.0)
        result = self.calculate(self.add_current(sender, recipient))

        self.assertEqual(result.sender.unique_recipient_count, 2)
        self.assertEqual(result.recipient.unique_sender_count, 1)
        self.assertEqual(result.patterns, [])

    def test_transaction_not_found_returns_404(self):
        with self.assertRaises(HTTPException) as error:
            transaction_network(f"{self.prefix}_missing", db=self.db)

        self.assertEqual(error.exception.status_code, 404)

    def test_endpoint_response_contains_network_analysis_structure(self):
        sender = f"{self.prefix}_sender"
        recipient = f"{self.prefix}_recipient"
        self.add_previous("previous", sender, recipient, 35.0)
        current = self.add_current(sender, recipient, 100.0)

        response = transaction_network(current.transaction_id, db=self.db)
        data = response.model_dump()

        self.assertEqual(data["transaction_id"], current.transaction_id)
        self.assertEqual(data["sender"]["outgoing_transaction_count"], 2)
        self.assertEqual(data["recipient"]["incoming_transaction_count"], 2)
        self.assertEqual(data["relationship"]["previous_transaction_count"], 1)
        self.assertEqual(data["relationship"]["relationship_status"], "EXISTING_RELATIONSHIP")
        self.assertIsInstance(data["patterns"], list)


if __name__ == "__main__":
    unittest.main()