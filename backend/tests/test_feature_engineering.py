import unittest

import pandas as pd

from app.services.feature_engineering import (
    FeatureEngineeringError,
    create_model_features,
)


FEATURE_LIST = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_change",
    "dest_balance_change",
    "amount_to_orig_balance",
    "amount_to_dest_balance",
    "orig_balance_error",
    "dest_balance_error",
    "log_amount",
    "hour",
    "day",
    "is_night",
    "hour_sin",
    "hour_cos",
    "user_tx_count_before",
    "user_amount_sum_before",
    "user_avg_amount_before",
    "amount_vs_user_avg",
    "user_max_amount_before",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


def transaction(**overrides):
    value = {
        "step": 49,
        "amount": 100.0,
        "type": "TRANSFER",
        "nameOrig": "C1",
        "oldbalanceOrg": 300.0,
        "newbalanceOrig": 200.0,
        "oldbalanceDest": 50.0,
        "newbalanceDest": 150.0,
    }
    value.update(overrides)
    return value


class FeatureEngineeringTests(unittest.TestCase):
    def test_builds_exact_order_and_notebook_formulas(self):
        history = pd.DataFrame(
            [
                {"step": 47, "amount": 20.0, "nameOrig": "C1"},
                {"step": 49, "amount": 30.0, "nameOrig": "C1"},
                {"step": 48, "amount": 900.0, "nameOrig": "C2"},
            ]
        )

        features = create_model_features(transaction(), history, FEATURE_LIST)

        self.assertEqual(list(features.columns), FEATURE_LIST)
        self.assertEqual(features.loc[0, "orig_balance_change"], 100.0)
        self.assertEqual(features.loc[0, "dest_balance_change"], 100.0)
        self.assertEqual(features.loc[0, "amount_to_orig_balance"], 100.0 / 301.0)
        self.assertEqual(features.loc[0, "amount_to_dest_balance"], 100.0 / 51.0)
        self.assertEqual(features.loc[0, "orig_balance_error"], 0.0)
        self.assertEqual(features.loc[0, "dest_balance_error"], 0.0)
        self.assertEqual(features.loc[0, "user_tx_count_before"], 2)
        self.assertEqual(features.loc[0, "user_amount_sum_before"], 50.0)
        self.assertEqual(features.loc[0, "user_avg_amount_before"], 25.0)
        self.assertEqual(features.loc[0, "amount_vs_user_avg"], 100.0 / 26.0)
        self.assertEqual(features.loc[0, "user_max_amount_before"], 30.0)
        self.assertEqual(features.loc[0, "type_TRANSFER"], 1)
        self.assertEqual(features.loc[0, "type_PAYMENT"], 0)

    def test_first_sender_transaction_uses_notebook_zero_defaults(self):
        features = create_model_features(
            transaction(type="PAYMENT"),
            pd.DataFrame(columns=["step", "amount", "nameOrig"]),
            FEATURE_LIST,
        )

        self.assertEqual(features.loc[0, "user_tx_count_before"], 0)
        self.assertEqual(features.loc[0, "user_amount_sum_before"], 0.0)
        self.assertEqual(features.loc[0, "user_avg_amount_before"], 0.0)
        self.assertEqual(features.loc[0, "amount_vs_user_avg"], 100.0)
        self.assertEqual(features.loc[0, "user_max_amount_before"], 0.0)

    def test_future_steps_are_rejected(self):
        history = pd.DataFrame(
            [{"step": 50, "amount": 10.0, "nameOrig": "C1"}]
        )

        with self.assertRaisesRegex(FeatureEngineeringError, "future-step"):
            create_model_features(transaction(step=49), history, FEATURE_LIST)

    def test_unsupported_type_is_rejected(self):
        with self.assertRaisesRegex(FeatureEngineeringError, "Unsupported transaction type"):
            create_model_features(
                transaction(type="UNKNOWN"),
                pd.DataFrame(columns=["step", "amount", "nameOrig"]),
                FEATURE_LIST,
            )

    def test_feature_list_mismatch_is_rejected(self):
        with self.assertRaisesRegex(FeatureEngineeringError, "feature list"):
            create_model_features(
                transaction(),
                pd.DataFrame(columns=["step", "amount", "nameOrig"]),
                FEATURE_LIST[:-1],
            )


if __name__ == "__main__":
    unittest.main()