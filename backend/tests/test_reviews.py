import unittest

from tests.helpers import ADMIN, ANALYST, ApiTestCase, unique_payload

STATUSES = ["OPEN", "UNDER_REVIEW", "DISMISSED", "ESCALATED", "CONFIRMED_SUSPICIOUS"]


class ReviewWorkflowTests(ApiTestCase):
    def new_transaction(self):
        response = self.analyze()
        self.assertEqual(response.status_code, 200)
        return response.json()

    def review_url(self, transaction_id):
        return f"/api/v1/reviews/{transaction_id}"

    def all_items(self, url):
        items = []
        for page in range(1, 200):
            response = self.client.get(f"{url}&page={page}&page_size=100", headers=ANALYST)
            self.assertEqual(response.status_code, 200, response.text)
            batch = response.json()["items"]
            items.extend(batch)
            if len(batch) < 100:
                break
        return items

    def test_create_review_defaults_to_open_and_records_reviewer(self):
        tx = self.new_transaction()
        response = self.client.post(self.review_url(tx["transaction_id"]), json={}, headers=ANALYST)
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertEqual(body["status"], "OPEN")
        self.assertEqual(body["reviewed_by"], "analyst")
        self.assertIsNotNone(body["reviewed_at"])
        self.assertIsNotNone(body["created_at"])
        self.assertIsNotNone(body["updated_at"])
        self.assertEqual(body["model_assessment"]["risk_level"], tx["risk_level"])

    def test_create_with_note_decision_and_user_label(self):
        tx = self.new_transaction()
        response = self.client.post(
            self.review_url(tx["transaction_id"]),
            json={
                "status": "UNDER_REVIEW",
                "analyst_note": "Contacted sender; awaiting reply.",
                "decision": "Hold pending verification",
            },
            headers={**ANALYST, "X-TrustX-User": "alice"},
        )
        body = response.json()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(body["status"], "UNDER_REVIEW")
        self.assertEqual(body["analyst_note"], "Contacted sender; awaiting reply.")
        self.assertEqual(body["decision"], "Hold pending verification")
        self.assertEqual(body["reviewed_by"], "analyst:alice")

    def test_every_status_can_be_set_via_patch(self):
        tx = self.new_transaction()
        url = self.review_url(tx["transaction_id"])
        self.client.post(url, json={}, headers=ANALYST)
        for status in STATUSES:
            with self.subTest(status=status):
                response = self.client.patch(url, json={"status": status}, headers=ANALYST)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["status"], status)

    def test_patch_updates_note_and_timestamps(self):
        tx = self.new_transaction()
        url = self.review_url(tx["transaction_id"])
        created = self.client.post(url, json={}, headers=ANALYST).json()
        updated = self.client.patch(
            url,
            json={"analyst_note": "Customer confirmed the payment.", "status": "DISMISSED"},
            headers={**ADMIN, "X-TrustX-User": "bob"},
        ).json()
        self.assertEqual(updated["analyst_note"], "Customer confirmed the payment.")
        self.assertEqual(updated["status"], "DISMISSED")
        self.assertEqual(updated["reviewed_by"], "admin:bob")
        self.assertEqual(updated["created_at"], created["created_at"])
        self.assertGreater(updated["updated_at"], created["updated_at"])
        self.assertGreaterEqual(updated["reviewed_at"], created["reviewed_at"])

    def test_patch_can_clear_the_note(self):
        tx = self.new_transaction()
        url = self.review_url(tx["transaction_id"])
        self.client.post(url, json={"analyst_note": "temp"}, headers=ANALYST)
        cleared = self.client.patch(url, json={"analyst_note": None}, headers=ANALYST).json()
        self.assertIsNone(cleared["analyst_note"])

    def test_review_never_changes_model_output(self):
        tx = self.new_transaction()
        before = self.client.get(
            f"/api/v1/transactions/{tx['transaction_id']}", headers=ANALYST
        ).json()
        url = self.review_url(tx["transaction_id"])
        self.client.post(url, json={"status": "CONFIRMED_SUSPICIOUS", "decision": "x"}, headers=ANALYST)
        self.client.patch(url, json={"status": "DISMISSED"}, headers=ANALYST)
        after = self.client.get(
            f"/api/v1/transactions/{tx['transaction_id']}", headers=ANALYST
        ).json()
        for field in ("fraud_probability", "risk_score", "risk_level", "recommended_action", "anomaly_signal"):
            self.assertEqual(before[field], after[field], field)
        self.assertIsNone(before["review_status"])
        self.assertEqual(after["review_status"], "DISMISSED")

    def test_duplicate_create_is_a_conflict(self):
        tx = self.new_transaction()
        url = self.review_url(tx["transaction_id"])
        self.assertEqual(self.client.post(url, json={}, headers=ANALYST).status_code, 201)
        self.assertEqual(self.client.post(url, json={}, headers=ANALYST).status_code, 409)

    def test_missing_transaction_and_missing_review(self):
        self.assertEqual(
            self.client.post("/api/v1/reviews/req_missing", json={}, headers=ANALYST).status_code, 404
        )
        tx = self.new_transaction()
        url = self.review_url(tx["transaction_id"])
        self.assertEqual(self.client.get(url, headers=ANALYST).status_code, 404)
        self.assertEqual(
            self.client.patch(url, json={"status": "OPEN"}, headers=ANALYST).status_code, 404
        )

    def test_invalid_review_input_is_rejected(self):
        tx = self.new_transaction()
        url = self.review_url(tx["transaction_id"])
        self.assertEqual(
            self.client.post(url, json={"status": "FRAUD"}, headers=ANALYST).status_code, 422
        )
        self.assertEqual(
            self.client.post(url, json={"analyst_note": "x" * 2001}, headers=ANALYST).status_code, 422
        )
        self.assertEqual(
            self.client.post(url, json={"reviewed_by": "someone"}, headers=ANALYST).status_code, 422
        )
        self.client.post(url, json={}, headers=ANALYST)
        self.assertEqual(self.client.patch(url, json={}, headers=ANALYST).status_code, 422)
        self.assertEqual(
            self.client.patch(url, json={"status": None}, headers=ANALYST).status_code, 422
        )

    def test_list_reviews_filters_by_status_and_risk(self):
        tx = self.new_transaction()
        url = self.review_url(tx["transaction_id"])
        self.client.post(url, json={"status": "ESCALATED"}, headers=ANALYST)
        listing = self.client.get("/api/v1/reviews?status=ESCALATED&page_size=100", headers=ANALYST)
        self.assertEqual(listing.status_code, 200)
        ids = [item["transaction_id"] for item in listing.json()["items"]]
        self.assertIn(tx["transaction_id"], ids)
        other = self.client.get("/api/v1/reviews?status=DISMISSED&page_size=100", headers=ANALYST)
        self.assertNotIn(tx["transaction_id"], [i["transaction_id"] for i in other.json()["items"]])
        level = tx["risk_level"]
        by_level = self.client.get(f"/api/v1/reviews?risk_level={level}&page_size=100", headers=ANALYST)
        self.assertIn(tx["transaction_id"], [i["transaction_id"] for i in by_level.json()["items"]])
        self.assertEqual(
            self.client.get("/api/v1/reviews?risk_level=EXTREME", headers=ANALYST).status_code, 422
        )

    def test_review_is_persisted_in_database(self):
        from app.db.models import CaseReview

        tx = self.new_transaction()
        self.client.post(
            self.review_url(tx["transaction_id"]),
            json={"status": "ESCALATED", "analyst_note": "n"},
            headers=ANALYST,
        )
        self.db.rollback()
        row = self.db.query(CaseReview).filter_by(transaction_id=tx["transaction_id"]).one()
        self.assertEqual((row.status, row.analyst_note, row.reviewed_by), ("ESCALATED", "n", "analyst"))

    def test_review_queue_prioritises_high_and_critical_without_closed_reviews(self):
        sender = f"RQ{self.new_transaction()['transaction_id'][4:14]}"
        high = self.persist_row(sender, sender + "D", 10, risk_level="HIGH", probability=0.7)
        critical = self.persist_row(sender, sender + "E", 11, risk_level="CRITICAL", probability=0.95)
        closed = self.persist_row(sender, sender + "F", 12, risk_level="CRITICAL", probability=0.99)
        low = self.persist_row(sender, sender + "G", 13, risk_level="LOW", probability=0.01)
        confirmed = self.persist_row(sender, sender + "H", 14, risk_level="HIGH", probability=0.8)
        escalated = self.persist_row(sender, sender + "I", 15, risk_level="HIGH", probability=0.75)
        self.client.post(self.review_url(closed.transaction_id), json={"status": "DISMISSED"}, headers=ANALYST)
        self.client.post(self.review_url(confirmed.transaction_id), json={"status": "CONFIRMED_SUSPICIOUS"}, headers=ANALYST)
        self.client.post(self.review_url(escalated.transaction_id), json={"status": "ESCALATED"}, headers=ANALYST)
        self.client.post(self.review_url(high.transaction_id), json={"status": "UNDER_REVIEW"}, headers=ANALYST)

        items = self.all_items("/api/v1/transactions?review_queue=true&sort_by=risk_priority")
        queue = {"items": items}
        ids = [item["transaction_id"] for item in items]
        self.assertIn(critical.transaction_id, ids)
        self.assertIn(high.transaction_id, ids)
        self.assertNotIn(closed.transaction_id, ids)
        self.assertNotIn(confirmed.transaction_id, ids)
        self.assertIn(escalated.transaction_id, ids)
        self.assertNotIn(low.transaction_id, ids)
        self.assertLess(ids.index(critical.transaction_id), ids.index(high.transaction_id))
        statuses = {item["transaction_id"]: item["review_status"] for item in queue["items"]}
        self.assertEqual(statuses[high.transaction_id], "UNDER_REVIEW")
        self.assertIsNone(statuses[critical.transaction_id])
        for item in queue["items"]:
            self.assertIn(item["risk_level"], ("HIGH", "CRITICAL"))

    def test_priority_sort_orders_levels(self):
        sender = f"PS{self.new_transaction()['transaction_id'][4:14]}"
        rows = {
            level: self.persist_row(sender, f"{sender}{level[:2]}", 20 + i, risk_level=level, probability=p)
            for i, (level, p) in enumerate(
                [("LOW", 0.01), ("CRITICAL", 0.9), ("MEDIUM", 0.4), ("HIGH", 0.7)]
            )
        }
        mine = {r.transaction_id for r in rows.values()}
        order = [
            item["risk_level"]
            for item in self.all_items("/api/v1/transactions?sort_by=risk_priority")
            if item["transaction_id"] in mine
        ]
        self.assertEqual(order, ["CRITICAL", "HIGH", "MEDIUM", "LOW"])


class AuditLogTests(ApiTestCase):
    def audit_actions(self, transaction_id):
        response = self.client.get(
            f"/api/v1/audit-logs?resource_id={transaction_id}&page_size=100", headers=ADMIN
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["items"]

    def test_analysis_view_and_review_events_are_logged(self):
        tx = self.analyze().json()
        transaction_id = tx["transaction_id"]
        self.client.get(f"/api/v1/transactions/{transaction_id}", headers=ANALYST)
        self.client.post(f"/api/v1/reviews/{transaction_id}", json={"status": "OPEN"}, headers=ANALYST)
        self.client.patch(
            f"/api/v1/reviews/{transaction_id}",
            json={"status": "ESCALATED", "analyst_note": "private note text"},
            headers={**ANALYST, "X-TrustX-User": "carol"},
        )
        entries = self.audit_actions(transaction_id)
        actions = [entry["action"] for entry in entries]
        for expected in ("TRANSACTION_ANALYZE", "TRANSACTION_VIEW", "REVIEW_CREATE", "REVIEW_UPDATE"):
            self.assertIn(expected, actions)
        update = next(entry for entry in entries if entry["action"] == "REVIEW_UPDATE")
        self.assertEqual(update["actor"], "analyst:carol")
        self.assertEqual(update["role"], "ANALYST")
        self.assertEqual(update["metadata"]["status"], "ESCALATED")
        self.assertEqual(update["metadata"]["previous_status"], "OPEN")
        self.assertIsNotNone(update["timestamp"])

    def test_audit_log_never_contains_secrets_or_note_text(self):
        tx = self.analyze().json()
        transaction_id = tx["transaction_id"]
        self.client.post(
            f"/api/v1/reviews/{transaction_id}",
            json={"analyst_note": "private note text"},
            headers=ANALYST,
        )
        text = self.client.get(
            f"/api/v1/audit-logs?resource_id={transaction_id}", headers=ADMIN
        ).text
        self.assertNotIn("test-analyst-key", text)
        self.assertNotIn("private note text", text)

    def test_duplicate_submission_is_logged(self):
        payload = unique_payload()
        first = self.analyze(payload).json()
        self.analyze(payload)
        actions = [e["action"] for e in self.audit_actions(first["transaction_id"])]
        self.assertIn("TRANSACTION_ANALYZE_DUPLICATE", actions)

    def test_audit_entries_are_persisted_in_database(self):
        from app.db.models import AuditLog

        tx = self.analyze().json()
        self.db.rollback()
        rows = self.db.query(AuditLog).filter_by(resource_id=tx["transaction_id"]).all()
        self.assertTrue(any(row.action == "TRANSACTION_ANALYZE" for row in rows))



if __name__ == "__main__":
    unittest.main()
