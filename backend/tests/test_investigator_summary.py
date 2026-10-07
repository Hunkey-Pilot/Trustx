import unittest

from app.services.investigator_summary import build_investigator_summary
from app.services.recommendations import build_recommendations


def analysis(**overrides):
    base = {
        "type": "TRANSFER",
        "amount": 12345.5,
        "fraud_probability": 0.9731,
        "risk_score": 97.31,
        "risk_level": "CRITICAL",
        "recommended_action": "TEMPORARY_HOLD / ESCALATE_FOR_REVIEW",
        "anomaly_signal": -0.0421,
        "explanation": {
            "status": "available",
            "top_features": [
                {"feature": "orig_balance_error", "value": 0.0, "shap_value": 5.1, "direction": "increases_fraud_prediction"},
                {"feature": "type_PAYMENT", "value": 0.0, "shap_value": -1.2, "direction": "decreases_fraud_prediction"},
            ],
        },
        "behavioral_signals": {
            "status": "available",
            "sender": {
                "transaction_count_1h": 4,
                "historical_average_amount": 100.0,
                "amount_to_historical_average": 123.5,
            },
            "recipient": {"unique_senders_24h": 6},
            "sender_recipient": {"is_new_recipient_for_sender": True},
        },
        "network_signals": {
            "status": "available",
            "relationship": {"relationship_status": "NEW_RELATIONSHIP"},
            "recipient": {"unique_sender_count": 6},
            "sender": {"unique_recipient_count": 2},
            "patterns": [{"pattern": "HIGH_RECIPIENT_CONNECTIVITY", "detected": True}],
        },
    }
    base.update(overrides)
    return base


class InvestigatorSummaryTests(unittest.TestCase):
    def test_summary_uses_only_verified_facts(self):
        summary = build_investigator_summary(analysis())
        text = summary.investigator_summary
        self.assertIn("CRITICAL", text)
        self.assertIn("0.9731", text)
        self.assertIn("uncalibrated", text)
        self.assertIn("orig_balance_error", text)
        self.assertIn("123.5 times", text)
        self.assertIn("first analyzed transaction between this sender and recipient", text)
        self.assertIn("high connectivity", text)
        self.assertIn("below zero", text)
        self.assertIn("Analyst review is recommended", text)

    def test_summary_never_asserts_fraud(self):
        for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            with self.subTest(level=level):
                summary = build_investigator_summary(analysis(risk_level=level))
                combined = (summary.investigator_summary + summary.disclaimer).lower()
                for forbidden in ("definitely", "is a fraud", "confirmed fraud", "mule account", "guilty"):
                    self.assertNotIn(forbidden, combined)
                self.assertIn("does not establish", summary.disclaimer)

    def test_low_risk_has_no_review_recommendation(self):
        summary = build_investigator_summary(
            analysis(risk_level="LOW", fraud_probability=0.001, behavioral_signals=None, network_signals=None)
        )
        self.assertEqual(summary.investigation_recommendations, [])
        self.assertIn("routine monitoring", summary.investigator_summary)

    def test_summary_is_deterministic(self):
        self.assertEqual(build_investigator_summary(analysis()), build_investigator_summary(analysis()))

    def test_missing_signals_do_not_break_or_invent_facts(self):
        summary = build_investigator_summary(
            analysis(explanation=None, behavioral_signals={"status": "unavailable"}, network_signals=None, anomaly_signal=0.1)
        )
        text = summary.investigator_summary
        self.assertNotIn("attributions", text)
        self.assertNotIn("connectivity", text)
        self.assertNotIn("below zero", text)

    def test_no_sender_history_is_stated_plainly(self):
        summary = build_investigator_summary(
            analysis(behavioral_signals={"status": "available", "sender": {"historical_average_amount": None}, "recipient": {}, "sender_recipient": {}})
        )
        self.assertIn("No earlier analyzed transactions from this sender", summary.investigator_summary)

    def test_model_output_and_recommendations_are_separate_fields(self):
        summary = build_investigator_summary(analysis())
        self.assertTrue(summary.investigation_recommendations)
        self.assertNotIn("Recipient verification", summary.investigator_summary)
        codes = [item.code for item in summary.investigation_recommendations]
        self.assertEqual(len(codes), len(set(codes)))


class RecommendationTests(unittest.TestCase):
    def codes(self, level, **kwargs):
        return {item.code for item in build_recommendations(level, **kwargs)}

    def test_levels(self):
        self.assertEqual(self.codes("LOW"), set())
        self.assertEqual(self.codes("MEDIUM"), {"ADDITIONAL_VERIFICATION"})
        self.assertEqual(self.codes("HIGH"), {"ANALYST_REVIEW", "ADDITIONAL_VERIFICATION"})
        self.assertEqual(
            self.codes("CRITICAL"),
            {"ANALYST_REVIEW", "ADDITIONAL_VERIFICATION", "TRANSACTION_DELAY"},
        )

    def test_signal_driven_actions(self):
        network = {"status": "available", "relationship": {"relationship_status": "NEW_RELATIONSHIP"}, "patterns": [{"pattern": "HIGH_SENDER_CONNECTIVITY", "detected": True}]}
        codes = self.codes("HIGH", network_signals=network)
        self.assertIn("RECIPIENT_VERIFICATION", codes)
        self.assertIn("ACCOUNT_VERIFICATION", codes)
        self.assertIn("TRANSACTION_LIMIT_REVIEW", self.codes("HIGH", amount_sensitive=True))

    def test_amount_note_does_not_claim_legitimacy(self):
        items = build_recommendations("HIGH", amount_sensitive=True)
        limit = next(item for item in items if item.code == "TRANSACTION_LIMIT_REVIEW")
        self.assertIn("would not by itself make the transaction legitimate", limit.rationale)


if __name__ == "__main__":
    unittest.main()
