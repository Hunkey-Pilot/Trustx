from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


TRANSACTION_TYPES = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)

REQUIRED_TRANSACTION_COLUMNS = (
    "step",
    "amount",
    "type",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "nameOrig",
)

HISTORICAL_COLUMNS = ("step", "amount", "nameOrig")


class FeatureEngineeringError(ValueError):
    """Raised when model features cannot be constructed safely."""


def create_model_features(
    transaction: Mapping[str, Any],
    historical_context: pd.DataFrame,
    feature_list: Sequence[str],
) -> pd.DataFrame:
    """Create one model-ready row using the notebook's feature formulas.

    Point-in-time policy (runtime):

    * history rows with ``step < current step`` are used;
    * history rows with ``step == current step`` are accepted but IGNORED,
      because the order of transactions inside one PaySim step (one hour) is
      not defined, so they cannot be proven to precede the transaction;
    * history rows with ``step > current step`` are rejected.

    Difference from training: the original notebook computed ``user_*``
    features with ``groupby(nameOrig).cumcount()`` over the step-sorted
    dataframe, which also counted earlier rows of the same step. The model
    artifacts are unchanged; in PaySim this only matters for the ~0.15% of
    rows whose sender appears more than once.
    """
    _validate_transaction(transaction)
    _validate_historical_context(historical_context, transaction["step"])

    step = transaction["step"]
    amount = transaction["amount"]
    oldbalance_org = transaction["oldbalanceOrg"]
    newbalance_orig = transaction["newbalanceOrig"]
    oldbalance_dest = transaction["oldbalanceDest"]
    newbalance_dest = transaction["newbalanceDest"]
    hour = step % 24

    sender_history = historical_context.loc[
        (historical_context["nameOrig"] == transaction["nameOrig"])
        & (historical_context["step"] < step)
    ].sort_values("step", kind="stable")

    user_tx_count_before = len(sender_history)
    user_amount_sum_before = float(sender_history["amount"].sum())
    user_avg_amount_before = (
        user_amount_sum_before / user_tx_count_before
        if user_tx_count_before > 0
        else 0.0
    )
    user_max_amount_before = (
        float(sender_history["amount"].max())
        if user_tx_count_before > 0
        else 0.0
    )

    transaction_type = transaction["type"]
    if transaction_type not in TRANSACTION_TYPES:
        raise FeatureEngineeringError(
            f"Unsupported transaction type: {transaction_type!r}"
        )

    generated_features: dict[str, Any] = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
        "orig_balance_change": oldbalance_org - newbalance_orig,
        "dest_balance_change": newbalance_dest - oldbalance_dest,
        "amount_to_orig_balance": amount / (oldbalance_org + 1),
        "amount_to_dest_balance": amount / (oldbalance_dest + 1),
        "orig_balance_error": oldbalance_org - amount - newbalance_orig,
        "dest_balance_error": oldbalance_dest + amount - newbalance_dest,
        "log_amount": np.log1p(amount),
        "hour": hour,
        "day": step // 24,
        "is_night": int(hour >= 0 and hour < 6),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "user_tx_count_before": user_tx_count_before,
        "user_amount_sum_before": user_amount_sum_before,
        "user_avg_amount_before": user_avg_amount_before,
        "amount_vs_user_avg": amount / (user_avg_amount_before + 1),
        "user_max_amount_before": user_max_amount_before,
        "type_CASH_IN": int(transaction_type == "CASH_IN"),
        "type_CASH_OUT": int(transaction_type == "CASH_OUT"),
        "type_DEBIT": int(transaction_type == "DEBIT"),
        "type_PAYMENT": int(transaction_type == "PAYMENT"),
        "type_TRANSFER": int(transaction_type == "TRANSFER"),
    }

    expected_features = list(feature_list)
    generated_features_set = set(generated_features)
    expected_features_set = set(expected_features)
    if len(expected_features) != 28:
        raise FeatureEngineeringError(
            f"Expected feature list to contain 28 names, got {len(expected_features)}"
        )
    if generated_features_set != expected_features_set:
        missing = sorted(expected_features_set - generated_features_set)
        unexpected = sorted(generated_features_set - expected_features_set)
        raise FeatureEngineeringError(
            f"Feature names do not match model feature list; missing={missing}, "
            f"unexpected={unexpected}"
        )

    features = pd.DataFrame(
        [[generated_features[name] for name in expected_features]],
        columns=expected_features,
    )
    _validate_feature_vector(features, expected_features)
    return features


def _validate_transaction(transaction: Mapping[str, Any]) -> None:
    missing = [
        column for column in REQUIRED_TRANSACTION_COLUMNS if column not in transaction
    ]
    if missing:
        raise FeatureEngineeringError(
            f"Transaction is missing required fields: {missing}"
        )

    numeric_fields = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    for field in numeric_fields:
        value = transaction[field]
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise FeatureEngineeringError(f"Transaction field {field!r} must be numeric")
        if not np.isfinite(value):
            raise FeatureEngineeringError(
                f"Transaction field {field!r} must be finite"
            )

    if transaction["amount"] < 0:
        raise FeatureEngineeringError("Transaction amount cannot be negative")
    if not isinstance(transaction["nameOrig"], str) or not transaction["nameOrig"]:
        raise FeatureEngineeringError("Transaction nameOrig must be a non-empty string")


def _validate_historical_context(
    historical_context: pd.DataFrame,
    current_step: int | float,
) -> None:
    if not isinstance(historical_context, pd.DataFrame):
        raise FeatureEngineeringError("historical_context must be a pandas DataFrame")

    missing = [
        column for column in HISTORICAL_COLUMNS if column not in historical_context.columns
    ]
    if missing:
        raise FeatureEngineeringError(
            f"Historical context is missing required columns: {missing}"
        )

    if historical_context.empty:
        return

    steps = pd.to_numeric(historical_context["step"], errors="coerce")
    amounts = pd.to_numeric(historical_context["amount"], errors="coerce")
    if steps.isna().any() or amounts.isna().any():
        raise FeatureEngineeringError(
            "Historical context step and amount values must be numeric and non-missing"
        )
    if not np.isfinite(steps.to_numpy()).all() or not np.isfinite(amounts.to_numpy()).all():
        raise FeatureEngineeringError(
            "Historical context step and amount values must be finite"
        )
    if (amounts < 0).any():
        raise FeatureEngineeringError(
            "Historical context amounts cannot be negative"
        )
    if historical_context["nameOrig"].isna().any():
        raise FeatureEngineeringError(
            "Historical context nameOrig values must not be missing"
        )
    if (steps > current_step).any():
        raise FeatureEngineeringError(
            "Historical context contains a future-step transaction"
        )


def _validate_feature_vector(
    features: pd.DataFrame,
    expected_features: list[str],
) -> None:
    if len(features.columns) != 28:
        raise FeatureEngineeringError(
            f"Expected 28 model features, got {len(features.columns)}"
        )
    if list(features.columns) != expected_features:
        raise FeatureEngineeringError("Feature names or order do not match feature_list.json")
    if features.isna().any().any():
        raise FeatureEngineeringError("Generated model features contain NaN values")

    numeric_values = features.to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise FeatureEngineeringError(
            "Generated model features contain infinite or non-numeric values"
        )