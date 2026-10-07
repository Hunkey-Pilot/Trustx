import os
import unittest
from unittest.mock import patch

from app.security import Actor, authenticate, require_admin
from tests.helpers import ADMIN, ADMIN_KEY, ANALYST, ANALYST_KEY, ApiTestCase, unique_payload


class ApiKeyAuthenticationTests(ApiTestCase):
    def test_health_is_public(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_protected_endpoints_require_an_api_key(self):
        protected = [
            ("GET", "/api/v1/transactions"),
            ("GET", "/api/v1/transactions/summary"),
            ("GET", "/api/v1/risk/summary"),
            ("GET", "/api/v1/transactions/req_x"),
            ("GET", "/api/v1/transactions/req_x/network"),
            ("GET", "/api/v1/transactions/req_x/counterfactual"),
            ("GET", "/api/v1/reviews"),
            ("GET", "/api/v1/reviews/req_x"),
            ("GET", "/api/v1/models/status"),
            ("GET", "/api/v1/audit-logs"),
        ]
        for method, path in protected:
            with self.subTest(path=path):
                response = self.client.request(method, path)
                self.assertEqual(response.status_code, 401, response.text)
        for method, path, body in [
            ("POST", "/api/v1/transactions/analyze", unique_payload()),
            ("POST", "/api/v1/reviews/req_x", {"status": "OPEN"}),
            ("PATCH", "/api/v1/reviews/req_x", {"status": "OPEN"}),
        ]:
            with self.subTest(path=path, method=method):
                self.assertEqual(self.client.request(method, path, json=body).status_code, 401)

    def test_invalid_api_key_is_rejected(self):
        response = self.client.get("/api/v1/transactions", headers={"X-API-Key": "wrong"})
        self.assertEqual(response.status_code, 401)
        self.assertNotIn(ANALYST_KEY, response.text)

    def test_analyst_key_can_analyze_and_read(self):
        analyzed = self.analyze(headers=ANALYST)
        self.assertEqual(analyzed.status_code, 200)
        transaction_id = analyzed.json()["transaction_id"]
        for path in (
            f"/api/v1/transactions/{transaction_id}",
            f"/api/v1/transactions/{transaction_id}/network",
            f"/api/v1/transactions/{transaction_id}/counterfactual",
            "/api/v1/transactions/summary",
            "/api/v1/risk/summary",
            "/api/v1/transactions?page_size=5",
            "/api/v1/reviews",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path, headers=ANALYST).status_code, 200)

    def test_admin_key_has_analyst_permissions_too(self):
        response = self.client.get("/api/v1/transactions/summary", headers=ADMIN)
        self.assertEqual(response.status_code, 200)

    def test_analyst_cannot_use_admin_endpoints(self):
        for path in ("/api/v1/models/status", "/api/v1/audit-logs"):
            with self.subTest(path=path):
                response = self.client.get(path, headers=ANALYST)
                self.assertEqual(response.status_code, 403)

    def test_admin_can_use_admin_endpoints(self):
        status_response = self.client.get("/api/v1/models/status", headers=ADMIN)
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["metadata"]["feature_count"], 28)
        self.assertEqual(self.client.get("/api/v1/audit-logs", headers=ADMIN).status_code, 200)

    def test_unconfigured_keys_fail_closed(self):
        with patch.dict(os.environ, {"TRUSTX_ANALYST_API_KEY": "", "TRUSTX_ADMIN_API_KEY": ""}):
            response = self.client.get("/api/v1/transactions", headers=ANALYST)
        self.assertEqual(response.status_code, 503)

    def test_role_without_configured_key_cannot_authenticate(self):
        with patch.dict(os.environ, {"TRUSTX_ADMIN_API_KEY": ""}):
            self.assertEqual(
                self.client.get("/api/v1/models/status", headers=ADMIN).status_code, 401
            )
            self.assertEqual(
                self.client.get("/api/v1/transactions/summary", headers=ANALYST).status_code, 200
            )

    def test_user_label_is_sanitised_and_attached_to_actor(self):
        with patch.dict(os.environ, {"TRUSTX_ANALYST_API_KEY": ANALYST_KEY}):
            actor = authenticate(x_api_key=ANALYST_KEY, x_trustx_user="  al<script>ice; DROP ")
        self.assertEqual(actor, Actor(name="analyst:alscripticeDROP", role="ANALYST"))

    def test_require_admin_rejects_analyst_actor(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as error:
            require_admin(Actor(name="analyst", role="ANALYST"))
        self.assertEqual(error.exception.status_code, 403)

    def test_cors_preflight_allows_api_key_header(self):
        response = self.client.options(
            "/api/v1/transactions/analyze",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-api-key,content-type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("x-api-key", response.headers["access-control-allow-headers"].lower())

    def test_rate_limit_returns_429(self):
        with patch.dict(os.environ, {"TRUSTX_RATE_LIMIT_PER_MINUTE": "2"}):
            codes = [self.analyze().status_code for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])


if __name__ == "__main__":
    unittest.main()
