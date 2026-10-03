from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from app.services.model_loader import LoadedModels


@dataclass(frozen=True)
class ShapFeatureContribution:
    feature: str
    value: float
    shap_value: float
    direction: str


@dataclass(frozen=True)
class ShapExplanation:
    status: str
    method: str
    explanation_type: str
    output_space: str | None
    base_value: float | None
    top_features: list[ShapFeatureContribution]
    error: str | None = None


class ShapService:
    """Generate optional SHAP model attributions for the trained XGBoost model."""

    def __init__(self, loaded_models: LoadedModels) -> None:
        self.feature_list = loaded_models.feature_list
        self.explainer: Any | None = None
        self.initialization_error: str | None = None
        try:
            self.explainer = shap.TreeExplainer(
                loaded_models.xgboost_model,
                model_output="raw",
            )
        except Exception as exc:
            self.initialization_error = f"{type(exc).__name__}: {exc}"

    def explain(self, features: pd.DataFrame, top_k: int = 5) -> ShapExplanation:
        if self.initialization_error is not None:
            return self._unavailable(self.initialization_error)

        try:
            self._validate_features(features)
            if self.explainer is None:
                raise RuntimeError("SHAP explainer is unavailable")

            shap_values = np.asarray(self.explainer.shap_values(features), dtype=float)
            if shap_values.shape != (1, len(self.feature_list)):
                raise RuntimeError(
                    "SHAP output shape does not match the 28-feature model vector"
                )
            if not np.isfinite(shap_values).all():
                raise RuntimeError("SHAP output contains NaN or infinite values")

            base_value = self._base_value()
            contributions = [
                ShapFeatureContribution(
                    feature=feature,
                    value=float(features.iloc[0][feature]),
                    shap_value=float(shap_value),
                    direction=direction_for_shap_value(float(shap_value)),
                )
                for feature, shap_value in zip(self.feature_list, shap_values[0])
            ]
            contributions.sort(key=lambda item: abs(item.shap_value), reverse=True)
            return ShapExplanation(
                status="available",
                method="SHAP",
                explanation_type="model_attribution",
                output_space="raw_margin",
                base_value=base_value,
                top_features=contributions[:top_k],
            )
        except Exception as exc:
            return self._unavailable(f"{type(exc).__name__}: {exc}")

    def _base_value(self) -> float:
        if self.explainer is None:
            raise RuntimeError("SHAP explainer is unavailable")
        base_value = np.asarray(self.explainer.expected_value, dtype=float).reshape(-1)
        if len(base_value) != 1 or not np.isfinite(base_value[0]):
            raise RuntimeError("SHAP base value is invalid")
        return float(base_value[0])

    def _validate_features(self, features: pd.DataFrame) -> None:
        if list(features.columns) != self.feature_list:
            raise ValueError("SHAP input feature names or order do not match feature_list.json")
        if features.shape != (1, len(self.feature_list)):
            raise ValueError("SHAP input must contain one row and exactly 28 features")
        values = features.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("SHAP input contains NaN or infinite values")

    @staticmethod
    def _unavailable(error: str) -> ShapExplanation:
        return ShapExplanation(
            status="unavailable",
            method="SHAP",
            explanation_type="model_attribution",
            output_space=None,
            base_value=None,
            top_features=[],
            error=error,
        )


def direction_for_shap_value(shap_value: float) -> str:
    if shap_value > 0:
        return "increases_fraud_prediction"
    if shap_value < 0:
        return "decreases_fraud_prediction"
    return "neutral"