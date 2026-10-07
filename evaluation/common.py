"""Shared helpers for the TrustX evaluation suite.

Everything here is READ-ONLY with respect to the production artifacts in
``Models/``. Ablation and baseline models are trained in memory and are never
written into ``Models/``.

All data are the PaySim SYNTHETIC dataset. Nothing in this package measures
real-world or real upay performance.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "Models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CSV = r"C:\Users\Anupam\Downloads\archive (3)\PS_20174392719_1491204439457_log.csv"
DEFAULT_CACHE = Path(os.getenv("TRUSTX_EVAL_CACHE", str(Path(os.getenv("TEMP", ".")) / "trustx_eval_cache")))

FEATURES: list[str] = json.loads((MODELS_DIR / "feature_list.json").read_text(encoding="utf-8"))
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

BALANCE_DERIVED = [
    "orig_balance_change", "dest_balance_change", "amount_to_orig_balance",
    "amount_to_dest_balance", "orig_balance_error", "dest_balance_error",
]
BALANCE_ERRORS = ["orig_balance_error", "dest_balance_error"]
TIME_FEATURES = ["step", "hour", "day", "is_night", "hour_sin", "hour_cos"]
HISTORY_FEATURES = [
    "user_tx_count_before", "user_amount_sum_before", "user_avg_amount_before",
    "amount_vs_user_avg", "user_max_amount_before",
]
TYPE_FEATURES = [f"type_{t}" for t in TYPES]

# Production risk bands (mirrors backend/app/services/risk_engine.py).
RISK_BANDS = [("LOW", 0.0, 0.30), ("MEDIUM", 0.30, 0.60), ("HIGH", 0.60, 0.85), ("CRITICAL", 0.85, 1.0 + 1e-9)]

# Production benchmark quoted in the README (strict temporal test set, threshold 0.5).
EXPECTED_BENCHMARK = {
    "precision": 1.000000, "recall": 0.999201, "f1": 0.999600,
    "roc_auc": 1.000000, "pr_auc": 0.999994,
    "tn": 88214, "fp": 0, "fn": 1, "tp": 1251,
}
FULL_DATASET_PREVALENCE = 8213 / 6362620


def load_paysim(path: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    csv = path or os.getenv("PAYSIM_CSV") or DEFAULT_CSV
    if not Path(csv).is_file():
        raise FileNotFoundError(
            f"PaySim CSV not found at {csv!r}. Pass --csv or set PAYSIM_CSV."
        )
    df = pd.read_csv(csv, nrows=nrows)
    df = df.drop(columns=["isFlaggedFraud"], errors="ignore")
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame, strict_same_step: bool = False) -> pd.DataFrame:
    """Vectorised version of backend/app/services/feature_engineering.py.

    ``strict_same_step=False`` reproduces the original notebook behaviour (earlier rows
    in step-sorted order, including same-step rows). ``True`` applies the runtime
    point-in-time policy (only strictly earlier steps). ``df`` must be sorted by step.
    """
    d = pd.DataFrame(index=df.index)
    d["step"] = df["step"].astype("int64")
    d["amount"] = df["amount"]
    d["oldbalanceOrg"] = df["oldbalanceOrg"]
    d["newbalanceOrig"] = df["newbalanceOrig"]
    d["oldbalanceDest"] = df["oldbalanceDest"]
    d["newbalanceDest"] = df["newbalanceDest"]
    d["orig_balance_change"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    d["dest_balance_change"] = df["newbalanceDest"] - df["oldbalanceDest"]
    d["amount_to_orig_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    d["amount_to_dest_balance"] = df["amount"] / (df["oldbalanceDest"] + 1)
    d["orig_balance_error"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    d["dest_balance_error"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    d["log_amount"] = np.log1p(df["amount"])
    d["hour"] = d["step"] % 24
    d["day"] = d["step"] // 24
    d["is_night"] = (d["hour"] < 6).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)

    if strict_same_step:
        count, total, maximum = _strict_history(df)
    else:
        grouped = df.groupby("nameOrig", sort=False)
        count = grouped.cumcount().to_numpy()
        total = (grouped["amount"].cumsum() - df["amount"]).to_numpy()
        prev_max = grouped["amount"].cummax().groupby(df["nameOrig"], sort=False).shift(1).fillna(0)
        maximum = np.where(count > 0, prev_max.to_numpy(), 0.0)
    d["user_tx_count_before"] = count
    d["user_amount_sum_before"] = total
    average = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    d["user_avg_amount_before"] = average
    d["amount_vs_user_avg"] = df["amount"].to_numpy() / (average + 1)
    d["user_max_amount_before"] = maximum
    for t in TYPES:
        d[f"type_{t}"] = (df["type"] == t).astype(int)
    return d[FEATURES]


def _strict_history(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """History using only rows with a strictly earlier step (exact, for repeat senders)."""
    n = len(df)
    count = np.zeros(n, dtype=np.int64)
    total = np.zeros(n, dtype=np.float64)
    maximum = np.zeros(n, dtype=np.float64)
    repeated = df[df.duplicated("nameOrig", keep=False)]
    for _, group in repeated.groupby("nameOrig", sort=False):
        steps = group["step"].to_numpy()
        amounts = group["amount"].to_numpy()
        positions = group.index.to_numpy()
        for i, position in enumerate(positions):
            earlier = steps < steps[i]
            if earlier.any():
                count[position] = int(earlier.sum())
                total[position] = float(amounts[earlier].sum())
                maximum[position] = float(amounts[earlier].max())
    return count, total, maximum


def strict_split(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The production model's strict temporal split (notebook cells 34-36)."""
    steps = np.sort(df["step"].unique())
    train_last = steps[int(len(steps) * 0.70) - 1]
    val_last = steps[int(len(steps) * 0.85) - 1]
    step = df["step"].to_numpy()
    return {
        "train": step <= train_last,
        "val": (step > train_last) & (step <= val_last),
        "test": step > val_last,
        "train_last_step": int(train_last),
        "val_last_step": int(val_last),
    }


def metrics(y_true, score, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    positives = int(y.sum())
    out: dict[str, Any] = {
        "n": int(len(y)),
        "positives": positives,
        "prevalence": positives / len(y) if len(y) else None,
        "threshold": threshold,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if 0 < positives < len(y):
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    else:
        out["roc_auc"] = out["pr_auc"] = None
    return out


def risk_level(probability: np.ndarray) -> np.ndarray:
    """Vectorised production risk bands on probability*100 (same boundaries as RiskEngine)."""
    score = np.round(np.asarray(probability) * 100, 6)
    return np.select([score < 30, score < 60, score < 85], ["LOW", "MEDIUM", "HIGH"], "CRITICAL")


def make_xgb(scale_pos_weight: float):
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.15, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        n_jobs=-1, random_state=42,
    )


def load_production_models():
    import joblib
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(MODELS_DIR / "final_xgb_model.json"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scikit-learn 1.6.1 -> 1.9.1 pickle warning
        isolation_forest = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    return booster, isolation_forest


def predict_production(booster, features: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb

    return booster.predict(xgb.DMatrix(features[FEATURES], feature_names=FEATURES))


def environment() -> dict[str, Any]:
    import sklearn
    import xgboost

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": " ".join(sys.argv),
    }


def save_json(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    payload = {"environment": environment(), **payload}
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def load_json(name: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value)}")


def guard_models_unchanged(before: dict[str, int]) -> None:
    """Fail loudly if an artifact in Models/ was modified during a run."""
    after = models_fingerprint()
    if before != after:
        raise RuntimeError("Models/ artifacts changed during evaluation - this must never happen")


def models_fingerprint() -> dict[str, int]:
    return {p.name: p.stat().st_mtime_ns for p in sorted(MODELS_DIR.iterdir()) if p.is_file()}
