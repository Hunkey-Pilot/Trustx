"""Point-in-time guarantees: a transaction's analysis may only use information
that existed strictly before it (earlier step AND already stored)."""

import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd

from app.db.models import TransactionAnalysis
from app.schemas.transactions import TransactionAnalyzeRequest
from app.services.behavioral_engine import BehavioralEngine
from app.services.feature_engineering import FeatureEngineeringError, create_model_features
from app.services.network_engine import NetworkEngine
from tests.helpers import ANALYST, ApiTestCase, unique_payload


FEATURES = None


def _feature_list():
    global FEATURES
    if FEATURES is None:
        import json
        from pathlib import Path

        FEATURES = json.loads(
            (Path(__file__).resolve().parents[2] / "Models" / "feature_list.json").read_text()
        )
    return FEATURES


def _transaction(step=100, sender="C1"):
    return {
        "step": step,
        "type": "TRANSFER",
        "amount": 100.0,
        "nameOrig": sender,
        "oldbalanceOrg": 300.0,
        "newbalanceOrig": 200.0,
        "nameDest": "M1",
        "oldbalanceDest": 50.0,
        "newbalanceDest": 150.0,
    }


def _history(*rows):
    return pd.DataFrame(rows, columns=["step", "amount", "nameOrig"])


class FeatureHistoryPolicyTests(unittest.TestCase):
    """Runtime feature history: past used, same-step ignored, future rejected."""

    def features(self, history, step=100):
        return create_model_features(_transaction(step), history, _feature_list())

    def test_past_transaction_is_used(self):
        row = self.features(_history((99, 40.0, "C1"))).iloc[0]
        self.assertEqual(row["user_tx_count_before"], 1)
        self.assertEqual(row["user_amount_sum_before"], 40.0)
        self.assertEqual(row["user_max_amount_before"], 40.0)

    def test_same_step_transaction_is_ignored(self):
        row = self.features(_history((100, 40.0, "C1"))).iloc[0]
        self.assertEqual(row["user_tx_count_before"], 0)
        self.assertEqual(row["user_amount_sum_before"], 0.0)
        self.assertEqual(row["user_avg_amount_before"], 0.0)
        self.assertEqual(row["user_max_amount_before"], 0.0)

    def test_same_step_row_does_not_change_features_of_past_history(self):
        past = self.features(_history((99, 40.0, "C1")))
        mixed = self.features(_history((99, 40.0, "C1"), (100, 9999.0, "C1")))
        pd.testing.assert_frame_equal(past, mixed)

    def test_future_transaction_is_rejected(self):
        with self.assertRaisesRegex(FeatureEngineeringError, "future-step"):
            self.features(_history((101, 40.0, "C1")))

    def test_future_transaction_is_rejected_even_next_to_valid_history(self):
        with self.assertRaisesRegex(FeatureEngineeringError, "future-step"):
            self.features(_history((50, 1.0, "C1"), (1000, 40.0, "C1")))

    def test_other_senders_never_contribute(self):
        row = self.features(_history((99, 40.0, "C2"))).iloc[0]
        self.assertEqual(row["user_tx_count_before"], 0)

    def test_features_do_not_depend_on_history_row_order(self):
        rows = [(10, 5.0, "C1"), (20, 7.0, "C1"), (30, 9.0, "C1")]
        forward = self.features(_history(*rows))
        backward = self.features(_history(*reversed(rows)))
        pd.testing.assert_frame_equal(forward, backward)


class PointInTimeDatabaseTests(ApiTestCase):
    def request_for(self, record):
        return TransactionAnalyzeRequest(
            step=record.step,
            type=record.transaction_type,
            amount=record.amount,
            nameOrig=record.name_orig,
            oldbalanceOrg=record.old_balance_org,
            newbalanceOrig=record.new_balance_orig,
            nameDest=record.name_dest,
            oldbalanceDest=record.old_balance_dest,
            newbalanceDest=record.new_balance_dest,
        )

    def network(self, current):
        return NetworkEngine().calculate(
            self.db,
            self.request_for(current),
            transaction_id=current.transaction_id,
            analysis_time=current.created_at,
        )

    def behavior(self, current):
        return BehavioralEngine().calculate(
            self.db,
            self.request_for(current),
            transaction_id=current.transaction_id,
            analysis_time=current.created_at,
        )

    def names(self):
        token = uuid4().hex[:10]
        return f"TS{token}", f"TD{token}"

    def test_past_step_row_is_used(self):
        sender, recipient = self.names()
        self.persist_row(sender, recipient, step=10)
        current = self.persist_row(sender, recipient, step=11, created_at=datetime.now(timezone.utc))
        network = self.network(current)
        self.assertEqual(network.relationship.previous_transaction_count, 1)
        self.assertEqual(network.sender.outgoing_transaction_count, 2)

    def test_future_step_row_is_excluded_even_if_stored_earlier(self):
        sender, recipient = self.names()
        # Stored BEFORE the current analysis but belongs to a later simulated step.
        self.persist_row(sender, recipient, step=101, amount=777.0)
        current = self.persist_row(sender, recipient, step=100, created_at=datetime.now(timezone.utc))
        network = self.network(current)
        self.assertEqual(network.relationship.previous_transaction_count, 0)
        self.assertEqual(network.relationship.relationship_status, "NEW_RELATIONSHIP")
        self.assertEqual(network.sender.outgoing_transaction_count, 1)
        self.assertEqual(network.recipient.total_incoming_amount, 100.0)
        behavior = self.behavior(current)
        self.assertEqual(behavior.sender.transaction_count_24h, 0)
        self.assertIsNone(behavior.sender.historical_max_amount)
        self.assertEqual(behavior.sender_recipient.pair_transaction_count, 0)

    def test_same_step_row_is_excluded(self):
        sender, recipient = self.names()
        self.persist_row(sender, recipient, step=100, amount=500.0)
        current = self.persist_row(sender, recipient, step=100, created_at=datetime.now(timezone.utc))
        self.assertEqual(self.network(current).relationship.previous_transaction_count, 0)
        self.assertEqual(self.behavior(current).sender.transaction_count_5m, 0)

    def test_row_stored_after_the_analysis_is_excluded(self):
        sender, recipient = self.names()
        now = datetime.now(timezone.utc)
        current = self.persist_row(sender, recipient, step=100, created_at=now - timedelta(minutes=10))
        # Earlier simulated step, but inserted later (back-filled): not known at analysis time.
        self.persist_row(sender, recipient, step=50, created_at=now)
        self.assertEqual(self.network(current).relationship.previous_transaction_count, 0)
        self.assertEqual(self.behavior(current).sender.transaction_count_24h, 0)

    def test_recomputation_is_stable_when_history_grows_afterwards(self):
        sender, recipient = self.names()
        now = datetime.now(timezone.utc)
        self.persist_row(sender, recipient, step=10, created_at=now - timedelta(minutes=30))
        current = self.persist_row(sender, recipient, step=100, created_at=now - timedelta(minutes=10))
        before = self.network(current).model_dump()
        self.persist_row(sender, recipient + "x", step=20, created_at=now)
        self.persist_row("other" + sender, recipient, step=30, created_at=now)
        self.assertEqual(self.network(current).model_dump(), before)

    def test_end_to_end_stored_network_is_not_altered_by_later_transactions(self):
        sender, recipient = self.names()
        first_payload = unique_payload(
            step=100, nameOrig=sender, nameDest=recipient
        )
        first = self.analyze(first_payload).json()
        stored_network = first["network_signals"]
        self.assertEqual(stored_network["source"], "analysis_time")
        self.assertEqual(stored_network["recipient"]["unique_sender_count"], 1)

        # Later analyses: an earlier-step transaction by another sender to the same
        # recipient, and a later-step one. Neither may alter the stored result.
        self.analyze(unique_payload(step=40, nameOrig="O1" + sender, nameDest=recipient))
        self.analyze(unique_payload(step=140, nameOrig="O2" + sender, nameDest=recipient))

        fetched = self.client.get(
            f"/api/v1/transactions/{first['transaction_id']}/network", headers=ANALYST
        ).json()
        self.assertEqual(fetched, stored_network)
        detail = self.client.get(
            f"/api/v1/transactions/{first['transaction_id']}", headers=ANALYST
        ).json()
        self.assertEqual(detail["network_signals"], stored_network)
        self.assertEqual(detail["behavioral_signals"], first["behavioral_signals"])

    def test_network_signals_are_persisted_at_analysis_time(self):
        response = self.analyze().json()
        self.db.rollback()
        row = self.db.query(TransactionAnalysis).filter_by(
            transaction_id=response["transaction_id"]
        ).one()
        self.assertEqual(row.network_signals["status"], "available")
        self.assertEqual(row.network_signals, response["network_signals"])
        self.assertEqual(row.behavioral_signals, response["behavioral_signals"])

    def test_later_step_analysis_sees_earlier_one_but_not_vice_versa(self):
        sender, recipient = self.names()
        late = self.analyze(unique_payload(step=200, nameOrig=sender, nameDest=recipient)).json()
        early = self.analyze(
            unique_payload(step=100, nameOrig=sender, nameDest=recipient, amount=55.0)
        ).json()
        self.assertEqual(late["network_signals"]["relationship"]["previous_transaction_count"], 0)
        self.assertEqual(early["network_signals"]["relationship"]["previous_transaction_count"], 0)
        later = self.analyze(
            unique_payload(step=300, nameOrig=sender, nameDest=recipient, amount=66.0)
        ).json()
        self.assertEqual(later["network_signals"]["relationship"]["previous_transaction_count"], 2)

    def test_legacy_row_is_recomputed_point_in_time_and_marked(self):
        sender, recipient = self.names()
        now = datetime.now(timezone.utc)
        self.persist_row(sender, recipient, step=10, created_at=now - timedelta(minutes=30))
        legacy = self.persist_row(
            sender, recipient, step=100, created_at=now - timedelta(minutes=10)
        )
        # Rows that must not influence the legacy recomputation.
        self.persist_row(sender, recipient, step=150, created_at=now - timedelta(minutes=20))
        self.persist_row(sender, recipient, step=20, created_at=now)
        self.assertIsNone(legacy.network_signals)

        network = self.client.get(
            f"/api/v1/transactions/{legacy.transaction_id}/network", headers=ANALYST
        ).json()
        self.assertEqual(network["source"], "legacy_recomputed")
        self.assertEqual(network["relationship"]["previous_transaction_count"], 1)
        self.assertEqual(network["sender"]["outgoing_transaction_count"], 2)

        behavior = self.client.get(
            f"/api/v1/transactions/{legacy.transaction_id}/behavior", headers=ANALYST
        ).json()
        self.assertEqual(behavior["source"], "legacy_recomputed")
        self.assertEqual(behavior["sender_recipient"]["pair_transaction_count"], 1)

    def test_stored_network_result_is_returned_verbatim(self):
        sender, recipient = self.names()
        stored = {
            "status": "available",
            "source": "analysis_time",
            "transaction_id": None,
            "sender": {"outgoing_transaction_count": 42, "unique_recipient_count": 7, "total_outgoing_amount": 1.5},
            "recipient": {"incoming_transaction_count": 3, "unique_sender_count": 2, "total_incoming_amount": 2.5},
            "relationship": {
                "previous_transaction_count": 9,
                "previous_transaction_amount": 3.5,
                "relationship_status": "EXISTING_RELATIONSHIP",
            },
            "patterns": [],
            "error": None,
        }
        row = self.persist_row(sender, recipient, step=5, network_signals=stored)
        fetched = self.client.get(
            f"/api/v1/transactions/{row.transaction_id}/network", headers=ANALYST
        ).json()
        self.assertEqual(fetched, stored)


if __name__ == "__main__":
    unittest.main()
