import unittest
from types import SimpleNamespace

from fastapi import HTTPException
from pydantic import ValidationError

from app.api.transactions import analyze_transaction
from app.main import app, model_loader
from app.schemas.transactions import TransactionAnalyzeRequest
from app.services.model_service import ModelService
from app.services.risk_engine import RiskEngine
from app.services.shap_service import ShapExplanation, ShapService
from app.services.evidence_engine import EvidenceEngine


BASE_TRANSACTION = {
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


class InferenceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loaded_models = model_loader.loaded or model_loader.load()
        app.state.model_loader = model_loader
        app.state.model_service = ModelService(loaded_models)
        app.state.risk_engine = RiskEngine()
        app.state.shap_service = ShapService(loaded_models)
        app.state.evidence_engine = EvidenceEngine()
        cls.request = SimpleNamespace(app=app)

    def analyze(self, payload):
        request = TransactionAnalyzeRequest.model_validate(payload)
        return analyze_transaction(self.request, request)

    def test_valid_transaction_uses_both_models(self):
        response = self.analyze({**BASE_TRANSACTION, "historical_transactions": []})

        self.assertIsInstance(response.fraud_probability, (int, float))
        self.assertIsInstance(response.anomaly_signal, (int, float))
        self.assertEqual(response.feature_count, 28)
        self.assertIsNone(response.model_version)
        self.assertEqual(response.explanation.status, "available")
        self.assertEqual(len(response.explanation.top_features), 5)
        self.assertEqual(response.evidence.status, "available")
        self.assertEqual(
            {item.category for item in response.evidence.items},
            {"MODEL_SIGNAL", "ANOMALY_SIGNAL", "MODEL_ATTRIBUTION"},
        )
        self.assertEqual(response.evidence.risk_assessment.risk_level, response.risk_level)
        self.assertEqual(response.fraud_probability, 0.00003663915776996873)

    def test_shap_integration_returns_sorted_saved_features_in_raw_margin_space(self):
        response = self.analyze({**BASE_TRANSACTION, "historical_transactions": []})
        explanation = response.explanation
        magnitudes = [abs(item.shap_value) for item in explanation.top_features]

        self.assertEqual(explanation.status, "available")
        self.assertEqual(explanation.output_space, "raw_margin")
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))
        self.assertTrue(
            set(item.feature for item in explanation.top_features).issubset(
                set(model_loader.loaded.feature_list)
            )
        )

    def test_shap_failure_does_not_break_prediction_response(self):
        original_service = app.state.shap_service
        failed_service = ShapService.__new__(ShapService)
        failed_service.feature_list = model_loader.loaded.feature_list
        failed_service.explainer = None
        failed_service.initialization_error = "simulated SHAP compatibility failure"
        app.state.shap_service = failed_service
        try:
            response = self.analyze({**BASE_TRANSACTION, "historical_transactions": []})
        finally:
            app.state.shap_service = original_service

        self.assertEqual(response.explanation.status, "unavailable")
        self.assertEqual(response.explanation.top_features, [])
        self.assertIsInstance(response.fraud_probability, (int, float))
        self.assertIsInstance(response.risk_score, (int, float))
        self.assertEqual(response.evidence.status, "partial")
        self.assertEqual(
            {item.category for item in response.evidence.items},
            {"MODEL_SIGNAL", "ANOMALY_SIGNAL"},
        )

    def test_sender_history_is_accepted(self):
        response = self.analyze(
            {
                **BASE_TRANSACTION,
                "historical_transactions": [
                    {"step": 47, "amount": 20.0, "nameOrig": "C1"},
                    {"step": 49, "amount": 30.0, "nameOrig": "C1"},
                ],
            }
        )

        self.assertEqual(response.feature_count, 28)

    def test_each_supported_type_is_accepted(self):
        for transaction_type in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
            with self.subTest(transaction_type=transaction_type):
                response = self.analyze({**BASE_TRANSACTION, "type": transaction_type})
                self.assertEqual(response.feature_count, 28)

    def test_missing_field_is_rejected(self):
        payload = {key: value for key, value in BASE_TRANSACTION.items() if key != "amount"}
        with self.assertRaises(ValidationError):
            TransactionAnalyzeRequest.model_validate(payload)

    def test_invalid_type_is_rejected(self):
        with self.assertRaises(ValidationError):
            TransactionAnalyzeRequest.model_validate(
                {**BASE_TRANSACTION, "type": "UNKNOWN"}
            )

    def test_non_finite_input_is_rejected(self):
        with self.assertRaises(ValidationError):
            TransactionAnalyzeRequest.model_validate(
                {**BASE_TRANSACTION, "amount": "NaN"}
            )

    def test_future_history_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            self.analyze(
                {
                    **BASE_TRANSACTION,
                    "historical_transactions": [
                        {"step": 50, "amount": 20.0, "nameOrig": "C1"}
                    ],
                }
            )
        self.assertEqual(error.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()