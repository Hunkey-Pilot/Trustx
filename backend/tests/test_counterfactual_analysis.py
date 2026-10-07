import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app.api.transactions import transaction_counterfactual
from app.main import app, model_loader
from app.schemas.transactions import TransactionAnalyzeRequest
from app.services.counterfactual_engine import CounterfactualEngine


class FakeModelService:
    def __init__(self, probabilities: dict[float, float]):
        self.probabilities = probabilities
        self.amounts: list[float] = []

    def predict(self, features):
        amount = float(features.iloc[0]["amount"])
        self.amounts.append(amount)
        return SimpleNamespace(fraud_probability=self.probabilities[amount])


class CounterfactualAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.transaction = TransactionAnalyzeRequest(
            step=49,
            type="TRANSFER",
            amount=100.0,
            nameOrig="CF_SENDER",
            oldbalanceOrg=300.0,
            newbalanceOrig=200.0,
            nameDest="CF_RECIPIENT",
            oldbalanceDest=50.0,
            newbalanceDest=150.0,
            historical_transactions=[
                {"step": 47, "amount": 20.0, "nameOrig": "CF_SENDER"},
                {"step": 48, "amount": 30.0, "nameOrig": "CF_SENDER"},
            ],
        )
        loaded = model_loader.loaded or model_loader.load()
        self.feature_list = loaded.feature_list

    def evaluate(self, model_service, current_probability=0.4, transaction=None):
        return CounterfactualEngine().evaluate(
            transaction_id="req_counterfactual_test",
            transaction=transaction or self.transaction,
            current_probability=current_probability,
            feature_list=self.feature_list,
            model_service=model_service,
        )

    def test_normal_transaction_has_four_valid_hypothetical_amounts(self):
        model = FakeModelService({100.0: 0.4, 75.0: 0.35, 50.0: 0.3, 25.0: 0.2})
        result = self.evaluate(model)

        self.assertEqual([item.amount_ratio for item in result.counterfactuals], [1.0, 0.75, 0.5, 0.25])
        self.assertEqual([item.amount for item in result.counterfactuals], [100.0, 75.0, 50.0, 25.0])
        self.assertTrue(all(item.valid for item in result.counterfactuals))
        self.assertEqual(model.amounts, [100.0, 75.0, 50.0, 25.0])

    def test_probability_and_probability_change_are_calculated(self):
        model = FakeModelService({100.0: 0.4, 75.0: 0.35, 50.0: 0.3, 25.0: 0.1})
        result = self.evaluate(model, current_probability=0.4)

        self.assertEqual(
            [item.fraud_probability for item in result.counterfactuals],
            [0.4, 0.35, 0.3, 0.1],
        )
        for scenario, expected_change in zip(
            result.counterfactuals,
            [0.0, -0.05, -0.1, -0.3],
        ):
            self.assertAlmostEqual(scenario.probability_change, expected_change)

    def test_best_counterfactual_is_largest_qualifying_reduction(self):
        model = FakeModelService({100.0: 0.72, 75.0: 0.55, 50.0: 0.31, 25.0: 0.18})
        result = self.evaluate(model, current_probability=0.72)

        self.assertEqual(result.best_counterfactual.amount, 25.0)
        self.assertEqual(result.best_counterfactual.amount_ratio, 0.25)
        self.assertAlmostEqual(result.best_counterfactual.probability_reduction, 0.54)
        self.assertIn("Hypothetical", result.best_counterfactual.description)

    def test_no_qualifying_counterfactual_returns_null_best(self):
        model = FakeModelService({100.0: 0.4, 75.0: 0.38, 50.0: 0.37, 25.0: 0.36})
        result = self.evaluate(model, current_probability=0.4)

        self.assertIsNone(result.best_counterfactual)

    def test_invalid_amounts_are_marked_without_model_inference(self):
        transaction = self.transaction.model_copy(
            update={
                "oldbalanceOrg": 30.0,
                "newbalanceOrig": 0.0,
            }
        )
        model = FakeModelService({25.0: 0.2})
        result = self.evaluate(model, current_probability=0.4, transaction=transaction)

        self.assertEqual([item.valid for item in result.counterfactuals], [False, False, False, True])
        self.assertEqual(model.amounts, [25.0])
        self.assertIsNone(result.counterfactuals[0].fraud_probability)

    def test_original_transaction_input_remains_unchanged(self):
        original = self.transaction.model_dump(mode="json")
        model = FakeModelService({100.0: 0.4, 75.0: 0.35, 50.0: 0.3, 25.0: 0.2})
        self.evaluate(model)

        self.assertEqual(self.transaction.model_dump(mode="json"), original)

    def test_transaction_not_found_returns_404(self):
        request = SimpleNamespace(app=app)
        with patch("app.api.transactions.get_transaction_analysis", return_value=None):
            with self.assertRaises(HTTPException) as error:
                transaction_counterfactual("not-found", request, db=object())

        self.assertEqual(error.exception.status_code, 404)

    def test_endpoint_response_structure_and_persisted_record_immutability(self):
        record = SimpleNamespace(
            transaction_id="req_counterfactual_api",
            step=self.transaction.step,
            transaction_type=self.transaction.type,
            amount=self.transaction.amount,
            name_orig=self.transaction.nameOrig,
            old_balance_org=self.transaction.oldbalanceOrg,
            new_balance_orig=self.transaction.newbalanceOrig,
            name_dest=self.transaction.nameDest,
            old_balance_dest=self.transaction.oldbalanceDest,
            new_balance_dest=self.transaction.newbalanceDest,
            historical_transactions=[
                item.model_dump() for item in self.transaction.historical_transactions
            ],
            fraud_probability=0.4,
        )
        original_record = (
            record.amount,
            record.new_balance_orig,
            record.new_balance_dest,
        )
        loaded = model_loader.loaded or model_loader.load()
        previous_state = {
            "model_loader": getattr(app.state, "model_loader", None),
            "model_service": getattr(app.state, "model_service", None),
            "counterfactual_engine": getattr(app.state, "counterfactual_engine", None),
        }
        app.state.model_loader = model_loader
        app.state.model_service = FakeModelService(
            {100.0: 0.4, 75.0: 0.35, 50.0: 0.3, 25.0: 0.1}
        )
        app.state.counterfactual_engine = CounterfactualEngine()
        request = SimpleNamespace(app=app)
        try:
            with patch(
                "app.api.transactions.get_transaction_analysis",
                return_value=record,
            ):
                response = transaction_counterfactual(
                    record.transaction_id,
                    request,
                    db=object(),
                )
        finally:
            for key, value in previous_state.items():
                if value is None:
                    if hasattr(app.state, key):
                        delattr(app.state, key)
                else:
                    setattr(app.state, key, value)

        payload = response.model_dump()
        self.assertEqual(
            set(payload),
            {
                "transaction_id",
                "original_amount",
                "original_fraud_probability",
                "baseline_fraud_probability",
                "baseline_source",
                "baseline_matches_stored",
                "counterfactuals",
                "best_counterfactual",
                "best_scenario",
                "interpretation",
            },
        )
        self.assertEqual(len(payload["counterfactuals"]), 4)
        self.assertEqual(
            (record.amount, record.new_balance_orig, record.new_balance_dest),
            original_record,
        )


if __name__ == "__main__":
    unittest.main()