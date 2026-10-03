import unittest
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.schemas.transactions import (
    NetworkPattern,
    NetworkSignals,
    NetworkSenderSignals,
    NetworkRecipientSignals,
    NetworkRelationshipSignals,
    TransactionAnalyzeRequest,
)
from app.services.evidence_engine import EvidenceEngine
from app.services.network_engine import NetworkEngine


class NetworkEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = NetworkEngine()
        self.transaction = TransactionAnalyzeRequest(
            step=50,
            type="TRANSFER",
            amount=250.0,
            nameOrig="CUST-01",
            oldbalanceOrg=1000.0,
            newbalanceOrig=750.0,
            nameDest="MERCHANT-01",
            oldbalanceDest=200.0,
            newbalanceDest=450.0,
            historical_transactions=[],
        )

    def test_network_metrics_are_available_for_history_based_analysis(self):
        db: Session = SessionLocal()
        try:
            result = self.engine.calculate(
                db,
                self.transaction,
                transaction_id="net_test_001",
                analysis_time=datetime.now(timezone.utc),
            )
            self.assertEqual(result.status, "available")
            self.assertEqual(result.transaction_id, "net_test_001")
            self.assertIsInstance(result.sender, NetworkSenderSignals)
            self.assertIsInstance(result.recipient, NetworkRecipientSignals)
            self.assertIsInstance(result.relationship, NetworkRelationshipSignals)
            self.assertGreaterEqual(result.sender.outgoing_transaction_count, 1)
            self.assertGreaterEqual(result.recipient.incoming_transaction_count, 1)
            self.assertEqual(result.relationship.relationship_status, "NEW_RELATIONSHIP")
        finally:
            db.close()

    def test_evidence_includes_network_signals_without_changing_model_outputs(self):
        network_signals = NetworkSignals(
            status="available",
            transaction_id="net_test_002",
            sender=NetworkSenderSignals(
                outgoing_transaction_count=3,
                unique_recipient_count=3,
                total_outgoing_amount=300.0,
            ),
            recipient=NetworkRecipientSignals(
                incoming_transaction_count=5,
                unique_sender_count=5,
                total_incoming_amount=500.0,
            ),
            relationship=NetworkRelationshipSignals(
                previous_transaction_count=2,
                previous_transaction_amount=200.0,
                relationship_status="EXISTING_RELATIONSHIP",
            ),
            patterns=[
                NetworkPattern(pattern="HIGH_RECIPIENT_CONNECTIVITY", detected=True)
            ],
        )
        result = EvidenceEngine().aggregate(
            fraud_probability=0.2,
            anomaly_signal=-0.1,
            shap_explanation=None,
            risk_result=None,
            behavioral_signals=None,
            network_signals=network_signals,
        )
        network_items = [item for item in result.items if item.category == "NETWORK_SIGNAL"]
        self.assertEqual(len(network_items), 1)
        self.assertIn("network-based signal", network_items[0].description.lower())
        self.assertIn("unique senders", network_items[0].description)
        self.assertGreaterEqual(len(result.items), 2)


if __name__ == "__main__":
    unittest.main()
