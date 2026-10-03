from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb


class ModelArtifactError(RuntimeError):
    """Raised when a required model artifact cannot be loaded or validated."""


@dataclass
class LoadedModels:
    xgboost_model: xgb.Booster
    isolation_forest: Any
    feature_list: list[str]
    model_config: dict[str, Any]
    shap_feature_importance: pd.DataFrame


class ModelLoader:
    """Loads and validates TrustX model artifacts once during application startup."""

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = artifacts_dir or Path(__file__).resolve().parents[3] / "Models"
        self.loaded: LoadedModels | None = None

    def load(self) -> LoadedModels:
        xgb_path = self._require_file("final_xgb_model.json")
        isolation_forest_path = self._require_file("isolation_forest.joblib")
        feature_list_path = self._require_file("feature_list.json")
        model_config_path = self._require_file("model_config.json")
        shap_importance_path = self._require_file("shap_feature_importance.csv")

        try:
            xgboost_model = xgb.Booster()
            xgboost_model.load_model(str(xgb_path))
        except Exception as exc:
            raise ModelArtifactError("Unable to load final_xgb_model.json") from exc

        try:
            isolation_forest = joblib.load(isolation_forest_path)
        except Exception as exc:
            raise ModelArtifactError("Unable to load isolation_forest.joblib") from exc

        feature_list = self._load_feature_list(feature_list_path)
        model_config = self._load_json_object(model_config_path, "model_config.json")

        try:
            shap_feature_importance = pd.read_csv(shap_importance_path)
        except Exception as exc:
            raise ModelArtifactError(
                "Unable to load shap_feature_importance.csv"
            ) from exc

        self._validate_contract(xgboost_model, feature_list, model_config)
        if shap_feature_importance.empty:
            raise ModelArtifactError("shap_feature_importance.csv contains no rows")

        self.loaded = LoadedModels(
            xgboost_model=xgboost_model,
            isolation_forest=isolation_forest,
            feature_list=feature_list,
            model_config=model_config,
            shap_feature_importance=shap_feature_importance,
        )
        return self.loaded

    def status(self) -> dict[str, Any]:
        loaded = self.loaded
        statuses = {
            "xgboost": "loaded" if loaded is not None else "not_loaded",
            "isolation_forest": "loaded" if loaded is not None else "not_loaded",
            "feature_list": "loaded" if loaded is not None else "not_loaded",
            "model_config": "loaded" if loaded is not None else "not_loaded",
            "shap_feature_importance": "loaded" if loaded is not None else "not_loaded",
        }

        response: dict[str, Any] = dict(statuses)
        if loaded is not None:
            response["metadata"] = {
                "xgboost_model_type": type(loaded.xgboost_model).__name__,
                "feature_count": len(loaded.feature_list),
                "feature_names": loaded.feature_list,
                "model_config": loaded.model_config,
                "shap_feature_importance_rows": len(loaded.shap_feature_importance),
            }
        return response

    def _require_file(self, filename: str) -> Path:
        path = self.artifacts_dir / filename
        if not path.is_file():
            raise ModelArtifactError(f"Required model artifact is missing: {filename}")
        return path

    @staticmethod
    def _load_feature_list(path: Path) -> list[str]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelArtifactError("Unable to load feature_list.json") from exc

        if not isinstance(value, list) or not value or not all(
            isinstance(feature, str) and feature for feature in value
        ):
            raise ModelArtifactError("feature_list.json must contain a non-empty string list")
        return value

    @staticmethod
    def _load_json_object(path: Path, filename: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelArtifactError(f"Unable to load {filename}") from exc

        if not isinstance(value, dict):
            raise ModelArtifactError(f"{filename} must contain a JSON object")
        return value

    @staticmethod
    def _validate_contract(
        xgboost_model: xgb.Booster,
        feature_list: list[str],
        model_config: dict[str, Any],
    ) -> None:
        if model_config.get("n_features") != len(feature_list):
            raise ModelArtifactError(
                "model_config.json feature count does not match feature_list.json"
            )

        model_features = xgboost_model.feature_names
        if model_features != feature_list:
            raise ModelArtifactError(
                "XGBoost feature names/order do not match feature_list.json"
            )