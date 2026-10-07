import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.schemas.transactions import TransactionAnalyzeRequest
from app.services.behavioral_engine import BehavioralEngine
from tests.helpers import ApiTestCase


class BehavioralEngineTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        token = uuid4().hex[:10]
        self.sender = f"BS{token}"
        self.recipient = f"BD{token}"
        self.now = datetime.now(timezone.utc)

    def calculate(self, step=100, amount=100.0, recipient=None, at=None):
        request = TransactionAnalyzeRequest(
            step=step,
            type="TRANSFER",
            amount=amount,
            nameOrig=self.sender,
            oldbalanceOrg=1000.0,
            newbalanceOrig=1000.0 - amount,
            nameDest=recipient or self.recipient,
            oldbalanceDest=0.0,
            newbalanceDest=amount,
        )
        return BehavioralEngine().calculate(
            self.db, request, transaction_id=f"tt_calc_{uuid4().hex}", analysis_time=at or self.now
        )

    def test_first_transaction_has_no_history(self):
        signals = self.calculate()
        self.assertEqual(signals.status, "available")
        self.assertEqual(signals.source, "analysis_time")
        self.assertEqual(signals.sender.transaction_count_24h, 0)
        self.assertIsNone(signals.sender.historical_average_amount)
        self.assertIsNone(signals.sender.amount_to_historical_average)
        self.assertTrue(signals.sender_recipient.is_new_recipient_for_sender)
        self.assertIsNone(signals.sender_recipient.minutes_since_previous_pair_transaction)
        self.assertEqual(signals.recipient.transaction_count, 0)

    def test_sender_velocity_windows(self):
        self.persist_row(self.sender, self.recipient, 10, created_at=self.now - timedelta(minutes=2), amount=100.0)
        self.persist_row(self.sender, self.recipient, 20, created_at=self.now - timedelta(minutes=30), amount=300.0)
        self.persist_row(self.sender, self.recipient, 30, created_at=self.now - timedelta(hours=5), amount=50.0)
        self.persist_row(self.sender, self.recipient, 40, created_at=self.now - timedelta(days=3), amount=10.0)
        sender = self.calculate().sender
        self.assertEqual(sender.transaction_count_5m, 1)
        self.assertEqual(sender.transaction_count_1h, 2)
        self.assertEqual(sender.transaction_count_24h, 3)
        self.assertEqual(sender.amount_sum_1h, 400.0)
        self.assertEqual(sender.amount_sum_24h, 450.0)

    def test_amount_ratios_use_prior_history(self):
        self.persist_row(self.sender, self.recipient, 10, created_at=self.now - timedelta(minutes=2), amount=100.0)
        self.persist_row(self.sender, self.recipient, 20, created_at=self.now - timedelta(minutes=3), amount=300.0)
        sender = self.calculate(amount=1000.0).sender
        self.assertEqual(sender.historical_average_amount, 200.0)
        self.assertEqual(sender.amount_to_historical_average, 5.0)
        self.assertEqual(sender.historical_max_amount, 300.0)
        self.assertAlmostEqual(sender.amount_to_historical_max, 1000.0 / 300.0)

    def test_pair_relationship_signals(self):
        self.persist_row(self.sender, self.recipient, 10, created_at=self.now - timedelta(minutes=6), amount=40.0)
        known = self.calculate().sender_recipient
        self.assertFalse(known.is_new_recipient_for_sender)
        self.assertEqual(known.pair_transaction_count, 1)
        self.assertEqual(known.pair_amount_sum, 40.0)
        self.assertAlmostEqual(known.minutes_since_previous_pair_transaction, 6.0, places=1)
        new = self.calculate(recipient=self.recipient + "N").sender_recipient
        self.assertTrue(new.is_new_recipient_for_sender)

    def test_recipient_activity_counts_distinct_senders(self):
        for index in range(3):
            self.persist_row(
                f"{self.sender}{index}", self.recipient, 10 + index,
                created_at=self.now - timedelta(minutes=10 + index), amount=20.0,
            )
        recipient = self.calculate().recipient
        self.assertEqual(recipient.transaction_count, 3)
        self.assertEqual(recipient.unique_senders, 3)
        self.assertEqual(recipient.unique_senders_1h, 3)
        self.assertEqual(recipient.unique_senders_24h, 3)
        self.assertEqual(recipient.amount_sum, 60.0)

    def test_other_senders_do_not_affect_sender_signals(self):
        self.persist_row(self.sender + "other", self.recipient, 10, created_at=self.now - timedelta(minutes=1))
        self.assertEqual(self.calculate().sender.transaction_count_24h, 0)

    def test_current_transaction_is_excluded_from_its_own_history(self):
        record = self.persist_row(self.sender, self.recipient, 100, created_at=self.now)
        request = TransactionAnalyzeRequest(
            step=100, type="TRANSFER", amount=100.0, nameOrig=self.sender,
            oldbalanceOrg=1000.0, newbalanceOrig=900.0, nameDest=self.recipient,
            oldbalanceDest=100.0, newbalanceDest=200.0,
        )
        signals = BehavioralEngine().calculate(
            self.db, request, transaction_id=record.transaction_id, analysis_time=self.now
        )
        self.assertEqual(signals.sender.transaction_count_24h, 0)

    def test_database_failure_returns_unavailable_without_internal_details(self):
        class BrokenSession:
            def execute(self, *args, **kwargs):
                raise RuntimeError("password=secret host=db.internal")

            scalar = execute

        request = TransactionAnalyzeRequest(
            step=1, type="TRANSFER", amount=1.0, nameOrig="C1", oldbalanceOrg=1.0,
            newbalanceOrig=0.0, nameDest="C2", oldbalanceDest=0.0, newbalanceDest=1.0,
        )
        signals = BehavioralEngine().calculate(BrokenSession(), request)
        self.assertEqual(signals.status, "unavailable")
        self.assertNotIn("secret", signals.error)
        self.assertNotIn("internal", signals.error)


if __name__ == "__main__":
    unittest.main()
