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
        # CASH_IN: the sender's balance RISES and the counter-party's FALLS.
        transaction = self.make_transaction(
            type="CASH_IN",
            oldbalanceOrg=100.0,
            newbalanceOrig=200.0,
            oldbalanceDest=500.0,
            newbalanceDest=400.0,
        )
        model = FakeModelService()
        self.evaluate(transaction, model)

        first = model.inputs[1].iloc[0]  # 75% scenario
        self.assertEqual(first["amount"], 75.0)
        self.assertEqual(first["newbalanceOrig"], 175.0)
        self.assertEqual(first["newbalanceDest"], 425.0)
        self.assertEqual(first["orig_balance_error"], -2 * 75.0)
        self.assertEqual(first["orig_balance_change"], -75.0)

    def test_balance_semantics_for_every_transaction_type(self):
        expectations = {
            # type: (new sender, new recipient) at 50% of an amount of 100,
            # with old balances of 300 (sender) and 500 (recipient)
            "TRANSFER": (250.0, 550.0),
            "CASH_OUT": (250.0, 550.0),
            "DEBIT": (250.0, 550.0),
            "PAYMENT": (250.0, 500.0),  # merchant balance is not tracked
            "CASH_IN": (350.0, 450.0),
        }
        for transaction_type, (sender, recipient) in expectations.items():
            with self.subTest(transaction_type=transaction_type):
                transaction = self.make_transaction(
                    type=transaction_type,
                    oldbalanceOrg=300.0,
                    oldbalanceDest=500.0,
                )
                model = FakeModelService()
                result = self.evaluate(transaction, model)
                scenario = model.inputs[2].iloc[0]  # 50% scenario
                self.assertEqual(scenario["amount"], 50.0)
                self.assertEqual(scenario["newbalanceOrig"], sender)
                self.assertEqual(scenario["newbalanceDest"], recipient)
                self.assertEqual(len(result.counterfactuals), 4)

    def test_real_xgboost_for_every_type_and_ratio_with_consistent_baseline(self):
        from app.services.feature_engineering import create_model_features

        model = ModelService(self.loaded_models)
        for transaction_type in ("TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"):
            with self.subTest(transaction_type=transaction_type):
                transaction = self.make_transaction(
                    type=transaction_type, oldbalanceOrg=500.0, oldbalanceDest=500.0
                )
                result = self.evaluate(transaction, model, current_probability=0.5)
                self.assertEqual(
                    [item.amount_ratio for item in result.counterfactuals], [1.0, 0.75, 0.5, 0.25]
                )
                self.assertTrue(all(item.valid for item in result.counterfactuals))
                self.assertTrue(
                    all(0.0 <= item.fraud_probability <= 1.0 for item in result.counterfactuals)
                )
                # Baseline equals an independent run of the exact feature pipeline.
                original = transaction.model_dump(exclude={"historical_transactions"})
                baseline_inputs = CounterfactualEngine._hypothetical_transaction(original, 100.0)
                history = __import__("pandas").DataFrame(
                    [item.model_dump() for item in transaction.historical_transactions],
                    columns=["step", "amount", "nameOrig"],
                )
                features = create_model_features(
                    baseline_inputs, history, self.loaded_models.feature_list
                )
                self.assertEqual(
                    result.baseline_fraud_probability, model.predict(features).fraud_probability
                )
                self.assertEqual(result.counterfactuals[0].probability_change, 0.0)
                self.assertEqual(result.baseline_source, "recomputed")

    def test_cash_in_invalid_when_counterparty_balance_would_go_negative(self):
        transaction = self.make_transaction(
            type="CASH_IN",
            oldbalanceOrg=100.0,
            oldbalanceDest=60.0,
        )
        result = self.evaluate(transaction, FakeModelService())
        self.assertEqual([item.valid for item in result.counterfactuals], [False, False, True, True])

    def test_changes_are_measured_against_recomputed_baseline(self):
        transaction = self.make_transaction()
        # Stored probability (0.9) differs from the recomputed 100% baseline (0.6).
        model = FakeModelService(
            lambda amount: {100.0: 0.6, 75.0: 0.5, 50.0: 0.4, 25.0: 0.3}[amount]
        )
        result = self.evaluate(transaction, model, current_probability=0.9)

        self.assertEqual(result.baseline_fraud_probability, 0.6)
        self.assertEqual(result.baseline_source, "recomputed")
        self.assertFalse(result.baseline_matches_stored)
        self.assertEqual(result.original_fraud_probability, 0.9)
        self.assertAlmostEqual(result.counterfactuals[0].probability_change, 0.0)
        self.assertAlmostEqual(result.counterfactuals[3].probability_change, -0.3)
        self.assertAlmostEqual(result.best_counterfactual.probability_reduction, 0.3)
        self.assertEqual(result.best_scenario, result.best_counterfactual)

    def test_baseline_matches_stored_when_model_is_consistent(self):
        result = self.evaluate(
            self.make_transaction(), FakeModelService(lambda amount: 0.2), current_probability=0.2
        )
        self.assertTrue(result.baseline_matches_stored)

    def test_interpretation_separates_model_output_from_recommendations(self):
        model = FakeModelService(
            lambda amount: {100.0: 0.9, 75.0: 0.8, 50.0: 0.6, 25.0: 0.2}[amount]
        )
        result = CounterfactualEngine(0.05).evaluate(
            transaction_id="req_test",
            transaction=self.make_transaction(),
            current_probability=0.9,
            feature_list=self.loaded_models.feature_list,
            model_service=model,
            risk_level="CRITICAL",
            network_signals={"status": "available", "relationship": {"relationship_status": "NEW_RELATIONSHIP"}, "patterns": []},
        )
        interpretation = result.interpretation
        self.assertTrue(interpretation.model_output.startswith("MODEL OUTPUT"))
        codes = {item.code for item in interpretation.investigation_recommendations}
        self.assertTrue(
            {"ANALYST_REVIEW", "ADDITIONAL_VERIFICATION", "TRANSACTION_DELAY", "RECIPIENT_VERIFICATION", "TRANSACTION_LIMIT_REVIEW"}
            <= codes
        )
        self.assertIn("does not make a transaction legitimate", interpretation.disclaimer)
        self.assertNotIn("proof", interpretation.model_output.lower())

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
