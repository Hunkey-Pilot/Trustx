import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api.transactions import transaction_counterfactual
from app.main import app, model_loader
from app.schemas.transactions import TransactionAnalyzeRequest
from app.services.counterfactual_engine import CounterfactualEngine
from app.services.model_service import ModelService


class FakeModelService:
    def __init__(self, probability_for_amount=None):
        self.probability_for_amount = probability_for_amount or (lambda amount: 0.2)
        self.inputs = []

    def predict(self, features):
        self.inputs.append(features.copy(deep=True))
        probability = self.probability_for_amount(float(features.iloc[0]["amount"]))
        return SimpleNamespace(fraud_probability=probability)


class CounterfactualEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded_models = model_loader.loaded or model_loader.load()

    def make_transaction(self, **overrides):
        values = {
            "step": 49,
            "type": "TRANSFER",
            "amount": 100.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 300.0,
            "newbalanceOrig": 200.0,
            "nameDest": "M1",
            "oldbalanceDest": 50.0,
            "newbalanceDest": 150.0,
            "historical_transactions": [
                {"step": 47, "amount": 20.0, "nameOrig": "C1"},
                {"step": 48, "amount": 30.0, "nameOrig": "C1"},
            ],
        }
        values.update(overrides)
        return TransactionAnalyzeRequest.model_validate(values)

    def evaluate(self, transaction, model_service, current_probability=0.8, threshold=0.05):
        return CounterfactualEngine(threshold).evaluate(
            transaction_id="req_test",
            transaction=transaction,
            current_probability=current_probability,
            feature_list=self.loaded_models.feature_list,
            model_service=model_service,
        )

    def test_normal_transaction_generates_four_amount_scenarios(self):
        transaction = self.make_transaction()
        model = FakeModelService()
        result = self.evaluate(transaction, model)

        self.assertEqual([item.amount_ratio for item in result.counterfactuals], [1.0, 0.75, 0.5, 0.25])
        self.assertTrue(all(item.valid for item in result.counterfactuals))
        self.assertEqual(len(model.inputs), 4)

    def test_original_transaction_is_not_modified(self):
        transaction = self.make_transaction()
        original = transaction.model_dump(mode="json")
        self.evaluate(transaction, FakeModelService())
        self.assertEqual(transaction.model_dump(mode="json"), original)

    def test_hypothetical_amount_and_transfer_balances_are_consistent(self):
        transaction = self.make_transaction()
        model = FakeModelService()
        self.evaluate(transaction, model)

        first = model.inputs[1].iloc[0]
        self.assertEqual(first["amount"], 75.0)
        self.assertEqual(first["newbalanceOrig"], 225.0)
        self.assertEqual(first["newbalanceDest"], 125.0)
        self.assertEqual(first["orig_balance_error"], 0.0)
        self.assertEqual(first["dest_balance_error"], 0.0)

    def test_cash_in_balance_direction_is_preserved(self):
        transaction = self.make_transaction(
            type="CASH_IN",
            oldbalanceOrg=100.0,
            newbalanceOrig=200.0,
            oldbalanceDest=50.0,
            newbalanceDest=50.0,
        )
        model = FakeModelService()
        self.evaluate(transaction, model)

        first = model.inputs[1].iloc[0]
        self.assertEqual(first["newbalanceOrig"], 25.0)
        self.assertEqual(first["newbalanceDest"], 125.0)

    def test_existing_xgboost_pipeline_returns_probability(self):
        transaction = self.make_transaction()
        model = ModelService(self.loaded_models)
        result = self.evaluate(transaction, model)

        self.assertEqual(len(result.counterfactuals), 4)
        self.assertTrue(
            all(
                item.fraud_probability is None
                or 0.0 <= item.fraud_probability <= 1.0
                for item in result.counterfactuals
            )
        )

    def test_best_counterfactual_is_largest_probability_reduction(self):
        transaction = self.make_transaction()
        model = FakeModelService(
            lambda amount: {100.0: 0.72, 75.0: 0.55, 50.0: 0.31, 25.0: 0.18}[amount]
        )
        result = self.evaluate(transaction, model, current_probability=0.72)

        self.assertEqual(result.best_counterfactual.amount, 25.0)
        self.assertEqual(result.best_counterfactual.amount_ratio, 0.25)
        self.assertAlmostEqual(result.best_counterfactual.probability_reduction, 0.54)
        self.assertIn("Hypothetical scenario", result.best_counterfactual.description)

    def test_no_significant_counterfactual_is_not_forced(self):
        transaction = self.make_transaction()
        model = FakeModelService(lambda amount: 0.78)
        result = self.evaluate(
            transaction,
            model,
            current_probability=0.8,
            threshold=0.05,
        )

        self.assertIsNone(result.best_counterfactual)

    def test_invalid_negative_balance_scenarios_are_skipped(self):
        transaction = self.make_transaction(
            amount=100.0,
            oldbalanceOrg=10.0,
            newbalanceOrig=0.0,
        )
        model = FakeModelService()
        result = self.evaluate(transaction, model)

        self.assertEqual(len(result.counterfactuals), 4)
        self.assertTrue(all(not item.valid for item in result.counterfactuals))
        self.assertEqual(model.inputs, [])

    def test_result_is_deterministic(self):
        transaction = self.make_transaction()
        first = self.evaluate(
            transaction,
            FakeModelService(lambda amount: amount / 1000),
        )
        second = self.evaluate(
            transaction,
            FakeModelService(lambda amount: amount / 1000),
        )
        self.assertEqual(first, second)

    def test_probabilities_and_feature_values_are_finite(self):
        transaction = self.make_transaction()
        model = FakeModelService()
        result = self.evaluate(transaction, model)

        self.assertTrue(
            all(
                item.fraud_probability is None or math.isfinite(item.fraud_probability)
                for item in result.counterfactuals
            )
        )
        self.assertTrue(
            all(math.isfinite(value) for frame in model.inputs for value in frame.to_numpy().ravel())
        )

    def test_counterfactual_api_endpoint_uses_existing_model_services(self):
        transaction = self.make_transaction()
        record = SimpleNamespace(
            transaction_id="req_api_test",
            step=transaction.step,
            transaction_type=transaction.type,
            amount=transaction.amount,
            name_orig=transaction.nameOrig,
            old_balance_org=transaction.oldbalanceOrg,
            new_balance_orig=transaction.newbalanceOrig,
            name_dest=transaction.nameDest,
            old_balance_dest=transaction.oldbalanceDest,
            new_balance_dest=transaction.newbalanceDest,
            historical_transactions=[item.model_dump() for item in transaction.historical_transactions],
            fraud_probability=0.8,
        )
        loaded = model_loader.loaded or model_loader.load()
        app.state.model_loader = model_loader
        app.state.model_service = ModelService(loaded)
        app.state.counterfactual_engine = CounterfactualEngine()
        request = SimpleNamespace(app=app)
        with patch("app.api.transactions.get_transaction_analysis", return_value=record):
            response = transaction_counterfactual("req_api_test", request, db=object())

        registered_paths = app.openapi()["paths"]
        self.assertIn("/api/v1/transactions/{transaction_id}/counterfactual", registered_paths)
        self.assertEqual(response.transaction_id, "req_api_test")
        self.assertEqual(response.original_amount, 100.0)
        self.assertEqual(len(response.counterfactuals), 4)
        self.assertTrue(
            all(
                item.fraud_probability is None or math.isfinite(item.fraud_probability)
                for item in response.counterfactuals
            )
        )


if __name__ == "__main__":
    unittest.main()
