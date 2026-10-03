import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.feature_engineering import create_model_features
from app.services.model_loader import ModelLoader
from app.services.shap_service import ShapService, direction_for_shap_value


class ShapServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded_models = ModelLoader().load()
        cls.service = ShapService(cls.loaded_models)
        cls.transaction = {
            "step": 49,
            "type": "TRANSFER",
            "amount": 100.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 300.0,
            "newbalanceOrig": 200.0,
            "nameDest": "M1",
            "oldbalanceDest": 50.0,
            "newbalanceDest": 150.0,
        }
        cls.features = create_model_features(
            cls.transaction,
            pd.DataFrame(columns=["step", "amount", "nameOrig"]),
            cls.loaded_models.feature_list,
        )

    def test_valid_transaction_gets_shap_explanation(self):
        result = self.service.explain(self.features)
        self.assertEqual(result.status, "available")
        self.assertEqual(result.method, "SHAP")
        self.assertEqual(result.explanation_type, "model_attribution")
        self.assertEqual(result.output_space, "raw_margin")
        self.assertEqual(len(result.top_features), 5)

    def test_all_28_feature_attributions_are_available(self):
        result = self.service.explain(self.features, top_k=28)
        self.assertEqual(len(result.top_features), 28)
        self.assertEqual(
            {item.feature for item in result.top_features},
            set(self.loaded_models.feature_list),
        )
        self.assertTrue(all(np.isfinite(item.shap_value) for item in result.top_features))

    def test_top_features_are_sorted_by_absolute_shap_value(self):
        result = self.service.explain(self.features, top_k=28)
        magnitudes = [abs(item.shap_value) for item in result.top_features]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_shap_direction_mapping(self):
        self.assertEqual(
            direction_for_shap_value(0.1), "increases_fraud_prediction"
        )
        self.assertEqual(
            direction_for_shap_value(-0.1), "decreases_fraud_prediction"
        )
        self.assertEqual(direction_for_shap_value(0.0), "neutral")

    def test_feature_names_match_saved_feature_list(self):
        result = self.service.explain(self.features, top_k=28)
        self.assertEqual(
            {item.feature for item in result.top_features},
            set(self.loaded_models.feature_list),
        )

    def test_identical_input_is_deterministic(self):
        first = self.service.explain(self.features, top_k=28)
        second = self.service.explain(self.features, top_k=28)
        self.assertEqual(first, second)

    def test_shap_failure_is_unavailable(self):
        service = ShapService.__new__(ShapService)
        service.feature_list = self.loaded_models.feature_list
        service.explainer = None
        service.initialization_error = "simulated compatibility failure"

        result = service.explain(self.features)

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.top_features, [])
        self.assertIn("simulated compatibility failure", result.error)


if __name__ == "__main__":
    unittest.main()