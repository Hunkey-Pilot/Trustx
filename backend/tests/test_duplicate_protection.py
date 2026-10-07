import os
import unittest
from unittest.mock import patch

from app.db.models import TransactionAnalysis
from app.db.repository import request_fingerprint
from app.schemas.transactions import TransactionAnalyzeRequest
from tests.helpers import ANALYST, ApiTestCase, unique_payload


class DuplicateProtectionTests(ApiTestCase):
    def count_rows(self, sender):
        self.db.rollback()
        return self.db.query(TransactionAnalysis).filter(TransactionAnalysis.name_orig == sender).count()

    def test_first_submission_is_a_new_analysis(self):
        payload = unique_payload()
        response = self.analyze(payload)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["is_duplicate"])
        self.assertEqual(self.count_rows(payload["nameOrig"]), 1)

    def test_identical_resubmission_returns_the_existing_analysis(self):
        payload = unique_payload()
        first = self.analyze(payload).json()
        second = self.analyze(payload).json()

        self.assertTrue(second["is_duplicate"])
        self.assertEqual(second["transaction_id"], first["transaction_id"])
        self.assertEqual(second["fraud_probability"], first["fraud_probability"])
        self.assertEqual(second["risk_level"], first["risk_level"])
        self.assertEqual(self.count_rows(payload["nameOrig"]), 1)

    def test_resubmission_does_not_pollute_network_history(self):
        payload = unique_payload()
        first = self.analyze(payload).json()
        for _ in range(3):
            self.analyze(payload)
        network = self.client.get(
            f"/api/v1/transactions/{first['transaction_id']}/network", headers=ANALYST
        ).json()
        self.assertEqual(network["sender"]["outgoing_transaction_count"], 1)
        self.assertEqual(self.count_rows(payload["nameOrig"]), 1)

    def test_slightly_different_transaction_is_a_new_analysis(self):
        payload = unique_payload()
        first = self.analyze(payload).json()
        for field, value in (
            ("amount", 101.0),
            ("step", 50),
            ("newbalanceOrig", 199.0),
            ("type", "CASH_OUT"),
            ("nameDest", payload["nameDest"] + "x"),
        ):
            with self.subTest(field=field):
                changed = self.analyze({**payload, field: value})
                self.assertEqual(changed.status_code, 200)
                body = changed.json()
                self.assertFalse(body["is_duplicate"])
                self.assertNotEqual(body["transaction_id"], first["transaction_id"])

    def test_history_payload_is_not_part_of_the_fingerprint(self):
        payload = unique_payload()
        first = self.analyze(payload).json()
        again = self.analyze(
            {
                **payload,
                "historical_transactions": [
                    {"step": 1, "amount": 5.0, "nameOrig": payload["nameOrig"]}
                ],
            }
        ).json()
        self.assertTrue(again["is_duplicate"])
        self.assertEqual(again["transaction_id"], first["transaction_id"])

    def test_window_of_zero_disables_protection(self):
        payload = unique_payload()
        with patch.dict(os.environ, {"TRUSTX_DUPLICATE_WINDOW_SECONDS": "0"}):
            first = self.analyze(payload).json()
            second = self.analyze(payload).json()
        self.assertNotEqual(first["transaction_id"], second["transaction_id"])
        self.assertFalse(second["is_duplicate"])

    def test_fingerprint_is_deterministic_and_field_sensitive(self):
        base = unique_payload()
        request = TransactionAnalyzeRequest.model_validate(base)
        same = TransactionAnalyzeRequest.model_validate(dict(reversed(list(base.items()))))
        other = TransactionAnalyzeRequest.model_validate({**base, "amount": 100.5})
        self.assertEqual(request_fingerprint(request), request_fingerprint(same))
        self.assertNotEqual(request_fingerprint(request), request_fingerprint(other))
        self.assertEqual(len(request_fingerprint(request)), 64)


if __name__ == "__main__":
    unittest.main()
