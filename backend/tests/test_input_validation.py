import json
import unittest
from unittest.mock import patch

from app.main import app
from tests.helpers import ADMIN, ANALYST, ApiTestCase, unique_payload


class InputValidationTests(ApiTestCase):
    def post(self, payload):
        return self.client.post("/api/v1/transactions/analyze", json=payload, headers=ANALYST)

    def assert_clean_error(self, response, expected_status):
        self.assertEqual(response.status_code, expected_status, response.text)
        body = response.text
        for forbidden in ("Traceback", ".py", "psycopg", "sqlalchemy", "postgresql://"):
            self.assertNotIn(forbidden, body)

    def test_valid_payload_is_accepted(self):
        self.assertEqual(self.analyze().status_code, 200)

    def test_oversized_sender_id_is_rejected(self):
        self.assert_clean_error(self.post(unique_payload(nameOrig="C" * 101)), 422)

    def test_oversized_recipient_id_is_rejected(self):
        self.assert_clean_error(self.post(unique_payload(nameDest="M" * 101)), 422)

    def test_identifier_with_unsafe_characters_is_rejected(self):
        self.assert_clean_error(self.post(unique_payload(nameOrig="C1; DROP TABLE x")), 422)
        self.assert_clean_error(self.post(unique_payload(nameDest="<script>")), 422)

    def test_empty_identifier_is_rejected(self):
        self.assert_clean_error(self.post(unique_payload(nameOrig="")), 422)

    def test_oversized_history_is_rejected(self):
        history = [{"step": 1, "amount": 1.0, "nameOrig": "C1"}] * 501
        self.assert_clean_error(self.post(unique_payload(historical_transactions=history)), 422)

    def test_history_at_the_limit_is_accepted(self):
        token = unique_payload()
        history = [
            {"step": i % 49, "amount": 1.0, "nameOrig": token["nameOrig"]} for i in range(500)
        ]
        token["historical_transactions"] = history
        response = self.analyze(token)
        self.assertEqual(response.status_code, 200, response.text)

    def test_invalid_step_is_rejected(self):
        for step in (-1, 100_001, 1.5, "abc"):
            with self.subTest(step=step):
                self.assert_clean_error(self.post(unique_payload(step=step)), 422)

    def test_invalid_amount_is_rejected(self):
        for amount in (-5, 1e13, "NaN", "Infinity", "x", None):
            with self.subTest(amount=amount):
                self.assert_clean_error(self.post(unique_payload(amount=amount)), 422)

    def test_invalid_balance_is_rejected(self):
        self.assert_clean_error(self.post(unique_payload(oldbalanceOrg=-1)), 422)
        self.assert_clean_error(self.post(unique_payload(newbalanceDest=1e13)), 422)

    def test_unsupported_type_and_unknown_fields_are_rejected(self):
        self.assert_clean_error(self.post(unique_payload(type="REFUND")), 422)
        self.assert_clean_error(self.post(unique_payload(extra_field=1)), 422)

    def test_future_history_is_rejected_with_422(self):
        payload = unique_payload()
        payload["historical_transactions"] = [
            {"step": payload["step"] + 1, "amount": 5.0, "nameOrig": payload["nameOrig"]}
        ]
        response = self.post(payload)
        self.assert_clean_error(response, 422)
        self.assertIn("future", response.json()["detail"].lower())

    def test_malformed_json_returns_400(self):
        response = self.client.post(
            "/api/v1/transactions/analyze",
            content=b"{not json",
            headers={**ANALYST, "Content-Type": "application/json"},
        )
        self.assert_clean_error(response, 400)

    def test_oversized_body_returns_413(self):
        padding = "x" * (300 * 1024)
        response = self.client.post(
            "/api/v1/transactions/analyze",
            content=json.dumps({"pad": padding}).encode(),
            headers={**ANALYST, "Content-Type": "application/json"},
        )
        self.assert_clean_error(response, 413)

    def test_oversized_transaction_id_in_path_is_rejected(self):
        for path in ("/api/v1/transactions/", "/api/v1/reviews/"):
            with self.subTest(path=path):
                response = self.client.get(path + "x" * 200, headers=ANALYST)
                self.assert_clean_error(response, 422)

    def test_unsafe_transaction_id_in_path_is_rejected(self):
        response = self.client.get("/api/v1/transactions/bad id!", headers=ANALYST)
        self.assertIn(response.status_code, (404, 422))

    def test_unknown_transaction_returns_404(self):
        for suffix in ("", "/network", "/counterfactual", "/behavior"):
            with self.subTest(suffix=suffix):
                response = self.client.get(
                    f"/api/v1/transactions/req_missing{suffix}", headers=ANALYST
                )
                self.assert_clean_error(response, 404)

    def test_invalid_list_filters_are_rejected(self):
        self.assert_clean_error(
            self.client.get("/api/v1/transactions?risk_level=EXTREME", headers=ANALYST), 422
        )
        self.assert_clean_error(
            self.client.get("/api/v1/transactions?page_size=1000", headers=ANALYST), 422
        )
        self.assert_clean_error(
            self.client.get(
                "/api/v1/transactions?min_risk_score=90&max_risk_score=10", headers=ANALYST
            ),
            422,
        )

    def test_unexpected_server_error_is_hidden(self):
        with patch.object(
            app.state.risk_engine, "calculate", side_effect=RuntimeError("secret internal path D:/x.py")
        ):
            response = self.post(unique_payload())
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Internal server error"})
        self.assertNotIn("secret", response.text)

    def test_error_responses_do_not_echo_api_key(self):
        response = self.client.post(
            "/api/v1/transactions/analyze",
            json={"bad": True},
            headers=ANALYST,
        )
        self.assertNotIn(ANALYST["X-API-Key"], response.text)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
