from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from app.services.model_loader import LoadedModels


class ModelInferenceError(RuntimeError):
    """Raised when model input validation or inference fails."""


@dataclass(frozen=True)
class ModelPrediction:
    fraud_probability: float
    anomaly_signal: float
    feature_count: int


class ModelService:
    """Runs predictions using models loaded during application startup."""

    def __init__(self, loaded_models: LoadedModels) -> None:
        self.loaded_models = loaded_models

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        expected_features = self.loaded_models.feature_list
        if list(features.columns) != expected_features:
            raise ModelInferenceError(
                "Model input feature names or order do not match feature_list.json"
            )
        if len(features.columns) != 28 or len(features) != 1:
            raise ModelInferenceError("Model input must contain exactly one row and 28 features")
        values = features.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ModelInferenceError("Model input contains NaN or infinite values")

        try:
            xgb_input = xgb.DMatrix(features, feature_names=expected_features)
            fraud_probability = float(self.loaded_models.xgboost_model.predict(xgb_input)[0])
            anomaly_signal = float(
                self.loaded_models.isolation_forest.decision_function(features)[0]
            )
        except Exception as exc:
            raise ModelInferenceError("Trained model inference failed") from exc

        if not np.isfinite(fraud_probability) or not np.isfinite(anomaly_signal):
            raise ModelInferenceError("Model output contains NaN or infinite values")

        return ModelPrediction(
            fraud_probability=fraud_probability,
            anomaly_signal=anomaly_signal,
            feature_count=len(expected_features),
        )