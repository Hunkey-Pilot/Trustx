import unittest

from app.services.evidence_engine import EvidenceEngine
from app.services.risk_engine import RiskResult
from app.services.shap_service import (
    ShapExplanation,
    ShapFeatureContribution,
)


class EvidenceEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = EvidenceEngine()
        self.risk = RiskResult(
            risk_score=40.0,
            risk_level="MEDIUM",
            recommended_action="ADDITIONAL_VERIFICATION / MONITOR",
        )
        self.shap = ShapExplanation(
            status="available",
            method="SHAP",
            explanation_type="model_attribution",
            output_space="raw_margin",
            base_value=0.5,
            top_features=[
                ShapFeatureContribution(
                    feature="positive_feature",
                    value=1.0,
                    shap_value=4.0,
                    direction="increases_fraud_prediction",
                ),
                ShapFeatureContribution(
                    feature="negative_feature",
                    value=2.0,
                    shap_value=-2.0,
                    direction="decreases_fraud_prediction",
                ),
            ],
        )

    def test_valid_evidence_contains_all_signal_categories(self):
        result = self.engine.aggregate(0.4, -0.03, self.shap, self.risk)
        categories = {item.category for item in result.items}
        self.assertEqual(result.status, "available")
        self.assertEqual(
            categories,
            {"MODEL_SIGNAL", "ANOMALY_SIGNAL", "MODEL_ATTRIBUTION"},
        )

    def test_model_signals_are_preserved(self):
        result = self.engine.aggregate(0.4, -0.03, self.shap, self.risk)
        values = {item.type: item.value for item in result.items}
        self.assertEqual(values["fraud_probability"], 0.4)
        self.assertEqual(values["isolation_forest_decision"], -0.03)

    def test_shap_evidence_is_sorted_and_preserves_direction(self):
        result = self.engine.aggregate(0.4, -0.03, self.shap, self.risk)
        shap_items = [item for item in result.items if item.category == "MODEL_ATTRIBUTION"]
        self.assertEqual([item.feature for item in shap_items], ["positive_feature", "negative_feature"])
        self.assertEqual(shap_items[0].direction, "increases_fraud_prediction")
        self.assertEqual(shap_items[1].direction, "decreases_fraud_prediction")
        self.assertIn("toward the fraud class", shap_items[0].description)
        self.assertIn("away from the fraud class", shap_items[1].description)

    def test_shap_failure_produces_partial_evidence_without_fabrication(self):
        unavailable = ShapExplanation(
            status="unavailable",
            method="SHAP",
            explanation_type="model_attribution",
            output_space=None,
            base_value=None,
            top_features=[],
            error="compatibility failure",
        )
        result = self.engine.aggregate(0.4, 0.02, unavailable, self.risk)
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.items), 2)
        self.assertFalse(any(item.category == "MODEL_ATTRIBUTION" for item in result.items))

    def test_evidence_is_deterministic(self):
        first = self.engine.aggregate(0.4, -0.03, self.shap, self.risk)
        second = self.engine.aggregate(0.4, -0.03, self.shap, self.risk)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()