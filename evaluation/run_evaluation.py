"""TrustX evaluation suite (PaySim SYNTHETIC data, strict temporal split).

Usage (from the repository root):

    backend\\.venv\\Scripts\\python.exe evaluation\\run_evaluation.py --sections all
    backend\\.venv\\Scripts\\python.exe evaluation\\run_evaluation.py --sections benchmark,risk
    backend\\.venv\\Scripts\\python.exe evaluation\\run_evaluation.py --sections ablation --variants B,C2
    backend\\.venv\\Scripts\\python.exe evaluation\\run_evaluation.py --quick      # smoke test on a subsample

Sections: benchmark, ablation, baselines, isolation_forest, risk, thresholds,
fairness, adversarial, signals, report.

Production artifacts in Models/ are only loaded, never written. Ablation and
baseline models are trained in memory with the production hyper-parameters.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

ALL_SECTIONS = [
    "benchmark", "ablation", "baselines", "isolation_forest", "risk",
    "thresholds", "fairness", "adversarial", "signals", "report",
]

ABLATIONS = {
    "A_retrain": ("All 28 features, retrained on CPU (control for the GPU-trained artifact)", []),
    "B": ("Without orig_balance_error", ["orig_balance_error"]),
    "C": ("Without both balance-error features", c.BALANCE_ERRORS),
    "C2": ("Without all six balance-derived features", c.BALANCE_DERIVED),
    "D": ("Transaction-type only (5 one-hot features)", None),
    "E": ("History / behavioural features only (5 user_* features)", None),
    "F": ("Without time features (step, hour, day, is_night, hour_sin, hour_cos)", c.TIME_FEATURES),
    "G": ("Without balance-derived AND time features", c.BALANCE_DERIVED + c.TIME_FEATURES),
}


class Context:
    """Lazily loaded data shared by sections."""

    def __init__(self, args):
        self.args = args
        self.cache = Path(args.cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        t0 = time.time()
        nrows = self.args.quick_rows if self.args.quick else None
        self.df = c.load_paysim(self.args.csv, nrows=nrows)
        self.y = self.df["isFraud"].to_numpy()
        self.split = c.strict_split(self.df)
        self.X = c.build_features(self.df)  # notebook behaviour (reproduces the artifact)
        self.test = self.split["test"]
        self.train = self.split["train"]
        self.val = self.split["val"]
        print(f"[data] {len(self.df):,} rows loaded and featurised in {time.time() - t0:.0f}s "
              f"(train {self.train.sum():,} / val {self.val.sum():,} / test {self.test.sum():,})", flush=True)
        self._loaded = True
        return self

    def proba_path(self, name: str) -> Path:
        suffix = "_quick" if self.args.quick else ""
        return self.cache / f"proba_{name}{suffix}.npy"

    def production_proba(self) -> np.ndarray:
        path = self.proba_path("production")
        if path.is_file():
            return np.load(path)
        booster, _ = c.load_production_models()
        proba = c.predict_production(booster, self.X[self.test])
        np.save(path, proba)
        return proba

    def split_summary(self) -> dict:
        d = self.df
        out = {"rows_total": int(len(d)), "dataset_prevalence": float(self.y.mean()),
               "train_last_step": self.split["train_last_step"], "val_last_step": self.split["val_last_step"]}
        for name in ("train", "val", "test"):
            mask = self.split[name]
            steps = d.loc[mask, "step"]
            n_steps = int(steps.nunique())
            fraud = int(self.y[mask].sum())
            out[name] = {
                "rows": int(mask.sum()), "fraud": fraud, "prevalence": fraud / int(mask.sum()),
                "steps": n_steps, "fraud_per_step": fraud / n_steps,
                "legit_per_step": (int(mask.sum()) - fraud) / n_steps,
            }
        return out


# ---------------------------------------------------------------- benchmark
def section_benchmark(ctx: Context) -> None:
    ctx.load()
    proba = ctx.production_proba()
    yt = ctx.y[ctx.test]
    result = c.metrics(yt, proba)
    expected = c.EXPECTED_BENCHMARK
    cm = result["confusion_matrix"]
    matches = {
        "precision": round(result["precision"], 6) == expected["precision"],
        "recall": round(result["recall"], 6) == expected["recall"],
        "f1": round(result["f1"], 6) == expected["f1"],
        "roc_auc": round(result["roc_auc"], 6) == expected["roc_auc"],
        "pr_auc": round(result["pr_auc"], 6) == expected["pr_auc"],
        "confusion_matrix": [cm["tn"], cm["fp"], cm["fn"], cm["tp"]]
        == [expected["tn"], expected["fp"], expected["fn"], expected["tp"]],
    }
    test = ctx.df[ctx.test]
    fraud = test[yt == 1]
    legit = test[yt == 0]
    obe = ctx.X.loc[ctx.test, "orig_balance_error"].to_numpy()
    shortcut = {
        "orig_balance_error_is_zero_share_fraud": float((np.abs(obe[yt == 1]) < 1e-6).mean()),
        "orig_balance_error_is_zero_share_legit": float((np.abs(obe[yt == 0]) < 1e-6).mean()),
    }
    # Strict point-in-time variant of the user_* features (runtime policy).
    strict = c.build_features(ctx.df, strict_same_step=True)
    differing = int((strict[c.HISTORY_FEATURES].to_numpy() != ctx.X[c.HISTORY_FEATURES].to_numpy()).any(axis=1).sum())
    strict_proba = c.predict_production(c.load_production_models()[0], strict[ctx.test])
    c.save_json("benchmark", {
        "dataset": "PaySim (synthetic)",
        "note": "Synthetic simulator benchmark. NOT real-world or upay production performance.",
        "split": ctx.split_summary(),
        "production_model_strict_test": result,
        "matches_readme_benchmark": matches,
        "all_match": all(matches.values()),
        "shortcut_check": shortcut,
        "strict_pit_history_variant": {
            "rows_whose_user_features_differ_from_notebook_policy": differing,
            "metrics": c.metrics(yt, strict_proba),
            "max_abs_probability_difference_vs_notebook_policy": float(np.abs(strict_proba - proba).max()),
        },
        "test_type_counts": {"all": test["type"].value_counts().to_dict(), "fraud": fraud["type"].value_counts().to_dict()},
        "test_rows_with_sender_history": int((ctx.X.loc[ctx.test, "user_tx_count_before"] > 0).sum()),
    })
    print("[benchmark] reproduced README benchmark:", all(matches.values()), flush=True)


# ---------------------------------------------------------------- ablation
def _fit_eval(ctx: Context, name: str, columns: list[str]) -> dict:
    t0 = time.time()
    Xtr = ctx.X.loc[ctx.train, columns].astype("float32")
    ytr = ctx.y[ctx.train]
    Xv = ctx.X.loc[ctx.val, columns].astype("float32")
    yv = ctx.y[ctx.val]
    Xt = ctx.X.loc[ctx.test, columns]
    yt = ctx.y[ctx.test]
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    model = c.make_xgb(spw)
    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    proba = model.predict_proba(Xt)[:, 1]
    np.save(ctx.proba_path(name), proba)
    result = c.metrics(yt, proba)
    result.update({"features_used": len(columns), "feature_names": columns,
                   "scale_pos_weight": spw, "train_seconds": round(time.time() - t0, 1)})
    return result


def section_ablation(ctx: Context, variants: list[str]) -> None:
    ctx.load()
    for name in variants:
        description, removed = ABLATIONS[name]
        if name == "D":
            columns = c.TYPE_FEATURES
        elif name == "E":
            columns = c.HISTORY_FEATURES
        else:
            columns = [f for f in c.FEATURES if f not in removed]
        print(f"[ablation] {name}: {description} ({len(columns)} features) ...", flush=True)
        result = _fit_eval(ctx, name, columns)
        result.update({"variant": name, "description": description,
                       "removed": None if removed is None else list(removed)})
        if name == "E":
            result["test_rows_with_any_history"] = int((ctx.X.loc[ctx.test, "user_tx_count_before"] > 0).sum())
            result["train_rows_with_any_history"] = int((ctx.X.loc[ctx.train, "user_tx_count_before"] > 0).sum())
            amount = ctx.df.loc[ctx.test, "amount"].to_numpy()
            ratio = ctx.X.loc[ctx.test, "amount_vs_user_avg"].to_numpy()
            result["share_of_test_rows_where_amount_vs_user_avg_equals_amount"] = float(np.isclose(ratio, amount).mean())
            result["note"] = (
                "Almost no PaySim sender appears twice, so user_tx_count_before, user_amount_sum_before, "
                "user_avg_amount_before and user_max_amount_before are constant 0 for nearly every row. The only "
                "remaining signal is amount_vs_user_avg = amount / (avg + 1), which equals the raw transaction "
                "amount whenever the sender has no history. Any skill above chance in this variant therefore comes "
                "from the raw amount, NOT from sender behaviour.")
        c.save_json(f"ablation_{name}", result)
        print(f"[ablation] {name}: F1={result['f1']:.6f} PR-AUC={result['pr_auc']:.6f} "
              f"P={result['precision']:.4f} R={result['recall']:.4f} cm={result['confusion_matrix']}", flush=True)


# ---------------------------------------------------------------- baselines
def section_baselines(ctx: Context) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ctx.load()
    rng = np.random.default_rng(42)
    Xtr_all = ctx.X.loc[ctx.train]
    ytr_all = ctx.y[ctx.train]
    Xt = ctx.X.loc[ctx.test]
    yt = ctx.y[ctx.test]
    results: dict[str, dict] = {}

    def subsample(max_legit: int):
        fraud_idx = np.flatnonzero(ytr_all == 1)
        legit_idx = np.flatnonzero(ytr_all == 0)
        legit_idx = rng.choice(legit_idx, size=min(max_legit, len(legit_idx)), replace=False)
        idx = np.sort(np.concatenate([fraud_idx, legit_idx]))
        return Xtr_all.iloc[idx], ytr_all[idx], len(legit_idx)

    # Logistic regression (log-scale-friendly via standard scaling; balanced class weights).
    Xs, ys, n_legit = subsample(1_000_000)
    t0 = time.time()
    logreg = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=300))
    logreg.fit(Xs.astype("float64"), ys)
    p = logreg.predict_proba(Xt)[:, 1]
    np.save(ctx.proba_path("logreg"), p)
    results["logistic_regression"] = {
        **c.metrics(yt, p), "train_rows": int(len(ys)), "train_legit_sampled": int(n_legit),
        "train_fraud": int(ys.sum()), "seconds": round(time.time() - t0, 1),
        "training_note": "all training fraud rows + a random sample of legitimate rows (seed 42)",
    }

    # Random forest (subsampled; class_weight=balanced_subsample).
    Xs, ys, n_legit = subsample(400_000)
    t0 = time.time()
    forest = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=2, n_jobs=-1,
                                    class_weight="balanced_subsample", random_state=42)
    forest.fit(Xs, ys)
    p = forest.predict_proba(Xt)[:, 1]
    np.save(ctx.proba_path("random_forest"), p)
    results["random_forest"] = {
        **c.metrics(yt, p), "train_rows": int(len(ys)), "train_legit_sampled": int(n_legit),
        "train_fraud": int(ys.sum()), "seconds": round(time.time() - t0, 1),
        "training_note": "all training fraud rows + a random sample of legitimate rows (seed 42); "
                         "150 trees, max_depth 14",
    }

    # Rule: TRANSFER / CASH_OUT and the sender account is emptied.
    def rule(frame):
        return (frame["type"].isin(["TRANSFER", "CASH_OUT"]) & (frame["newbalanceOrig"] == 0)
                & (frame["oldbalanceOrg"] > 0)).astype(float).to_numpy()

    test_raw = ctx.df[ctx.test]
    results["rule_transfer_cashout_account_emptied"] = {
        **c.metrics(yt, rule(test_raw)),
        "definition": "type in (TRANSFER, CASH_OUT) AND newbalanceOrig == 0 AND oldbalanceOrg > 0",
    }

    # Amount threshold chosen on the TRAIN split only (maximise F1).
    train_amount = ctx.df.loc[ctx.train, "amount"].to_numpy()
    candidates = np.unique(np.quantile(train_amount, np.linspace(0.5, 0.9999, 400)))
    best_t, best_f1 = candidates[0], -1.0
    for threshold in candidates:
        pred = train_amount >= threshold
        tp = int((pred & (ytr_all == 1)).sum())
        fp = int((pred & (ytr_all == 0)).sum())
        fn = int((~pred & (ytr_all == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_t, best_f1 = float(threshold), f1
    results["amount_threshold"] = {
        **c.metrics(yt, (test_raw["amount"].to_numpy() >= best_t).astype(float)),
        "threshold_amount": best_t, "threshold_selected_on": "train split (max F1)", "train_f1": best_f1,
        "score_note": "ROC/PR-AUC use the binary flag; see amount_score_auc for the continuous amount",
        "amount_score_auc": {"roc_auc": c.metrics(yt, test_raw["amount"].to_numpy(), best_t)["roc_auc"],
                             "pr_auc": c.metrics(yt, test_raw["amount"].to_numpy(), best_t)["pr_auc"]},
    }
    c.save_json("baselines", {"models": results, "test": ctx.split_summary()["test"]})
    for name, r in results.items():
        print(f"[baselines] {name}: P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
              f"PR-AUC={r['pr_auc']:.4f}", flush=True)


# ---------------------------------------------------------------- isolation forest
def section_isolation_forest(ctx: Context) -> None:
    ctx.load()
    _, iso = c.load_production_models()
    Xt = ctx.X.loc[ctx.test]
    yt = ctx.y[ctx.test]
    decision = iso.decision_function(Xt[c.FEATURES])
    predicted = iso.predict(Xt[c.FEATURES])
    anomaly_score = -decision
    quantiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    rank_metrics = c.metrics(yt, anomaly_score, threshold=float(np.quantile(anomaly_score, 0.99)))
    native = c.metrics(yt, (predicted == -1).astype(float))
    c.save_json("isolation_forest", {
        "note": ("Isolation Forest is an anomaly SIGNAL. It is NOT part of the production risk score because no "
                 "production-safe calibration exists. No min-max normalisation from test-set statistics is used."),
        "ranking_quality_of_negative_decision_function": {"roc_auc": rank_metrics["roc_auc"], "pr_auc": rank_metrics["pr_auc"],
                                                          "prevalence": rank_metrics["prevalence"]},
        "native_predict_minus_one_as_flag": native,
        "decision_function_quantiles": {
            "quantile_levels": quantiles,
            "fraud": np.quantile(decision[yt == 1], quantiles).tolist(),
            "legit": np.quantile(decision[yt == 0], quantiles).tolist(),
        },
        "decision_function_mean": {"fraud": float(decision[yt == 1].mean()), "legit": float(decision[yt == 0].mean())},
        "share_below_zero": {"fraud": float((decision[yt == 1] < 0).mean()), "legit": float((decision[yt == 0] < 0).mean())},
        "trained_on": "normal rows of the older row-order 70% split (steps <= 323) per the training notebook",
        "sklearn_pickle_note": "Artifact was pickled with scikit-learn 1.6.1; evaluation loads it under the installed version.",
        "test": ctx.split_summary()["test"],
    })
    print(f"[isolation_forest] ROC-AUC={rank_metrics['roc_auc']:.4f} PR-AUC={rank_metrics['pr_auc']:.4f}", flush=True)


# ---------------------------------------------------------------- risk
def _risk_table(yt: np.ndarray, proba: np.ndarray) -> dict:
    levels = c.risk_level(proba)
    table = {}
    for level, _, _ in c.RISK_BANDS:
        m = levels == level
        table[level] = {"count": int(m.sum()), "fraud": int(yt[m].sum()),
                        "fraud_rate": float(yt[m].mean()) if m.any() else None}
    queue = (levels == "HIGH") | (levels == "CRITICAL")
    n, fraud_total = len(yt), int(yt.sum())
    tp = int(yt[queue].sum())
    fp = int(queue.sum()) - tp
    legit_total = n - fraud_total
    pi = c.FULL_DATASET_PREVALENCE
    # Re-weight legit rows so prevalence matches the full PaySim dataset (0.129%).
    legit_weight = (fraud_total * (1 - pi) / pi) / legit_total
    weighted_total = fraud_total + legit_total * legit_weight
    return {
        "levels": table,
        "high_queue": table["HIGH"]["count"], "critical_queue": table["CRITICAL"]["count"],
        "review_queue_size": int(queue.sum()),
        "review_queue_precision": tp / queue.sum() if queue.any() else None,
        "review_queue_recall": tp / fraud_total if fraud_total else None,
        "alerts_per_100k_at_test_prevalence": float(queue.sum() / n * 1e5),
        "alerts_per_100k_reweighted_to_full_dataset_prevalence": float((tp + fp * legit_weight) / weighted_total * 1e5),
        "reweighting": {"target_prevalence": pi, "legit_weight": legit_weight,
                        "method": "legitimate rows weighted so that fraud prevalence equals the full PaySim prevalence"},
        "false_positives_in_queue": fp,
        "fpr_rule_of_three_95pct_upper_bound": (3 / legit_total) if fp == 0 else None,
        "rows_with_probability_between_0.05_and_0.95": {
            "fraud": int(((proba > 0.05) & (proba < 0.95) & (yt == 1)).sum()),
            "legit": int(((proba > 0.05) & (proba < 0.95) & (yt == 0)).sum()),
        },
        "probability_histogram": {
            "bin_edges": [0, 0.01, 0.05, 0.3, 0.6, 0.85, 0.95, 0.99, 1.0],
            "fraud": np.histogram(proba[yt == 1], bins=[0, 0.01, 0.05, 0.3, 0.6, 0.85, 0.95, 0.99, 1.0000001])[0].tolist(),
            "legit": np.histogram(proba[yt == 0], bins=[0, 0.01, 0.05, 0.3, 0.6, 0.85, 0.95, 0.99, 1.0000001])[0].tolist(),
        },
    }


def section_risk(ctx: Context) -> None:
    ctx.load()
    yt = ctx.y[ctx.test]
    out = {"note": "Production thresholds: LOW <30, MEDIUM 30-59, HIGH 60-84, CRITICAL >=85 on probability*100. "
                   "HIGH and CRITICAL require human review. Counts are computed, not hard-coded.",
           "production_model": _risk_table(yt, ctx.production_proba())}
    for name in ("B", "C", "C2", "F", "G"):
        path = ctx.proba_path(name)
        if path.is_file():
            out[f"ablation_{name}"] = _risk_table(yt, np.load(path))
    if not (ctx.production_proba() > 0.3).any():
        out["warning"] = "production model produced no rows above 0.3"
    c.save_json("risk", out)
    prod = out["production_model"]
    print("[risk] production levels:", {k: v["count"] for k, v in prod["levels"].items()},
          "queue precision", prod["review_queue_precision"], flush=True)


# ---------------------------------------------------------------- thresholds
def _policy_cost(yt, amount, proba, t_review, t_hold, missed_mult, hold_cost, review_cost, legit_weight):
    review = proba >= t_review
    hold = proba >= t_hold
    fraud = yt == 1
    missed = fraud & ~review
    cost_missed = float((amount[missed]).sum() * missed_mult)
    cost_review = float(review_cost * (review[fraud].sum() + legit_weight * review[~fraud].sum()))
    false_hold = (hold & ~fraud)
    cost_hold = float(hold_cost * legit_weight * false_hold.sum())
    total = cost_missed + cost_review + cost_hold
    n_equiv = fraud.sum() + legit_weight * (~fraud).sum()
    return {"total_cost": total, "cost_per_100k": total / n_equiv * 1e5, "missed_fraud": int(missed.sum()),
            "reviewed": float(review[fraud].sum() + legit_weight * review[~fraud].sum()),
            "false_holds": float(legit_weight * false_hold.sum())}


def section_thresholds(ctx: Context) -> None:
    ctx.load()
    yt = ctx.y[ctx.test]
    amount = ctx.df.loc[ctx.test, "amount"].to_numpy()
    grid = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
    pi = c.FULL_DATASET_PREVALENCE
    fraud_n, legit_n = int(yt.sum()), int((yt == 0).sum())
    weights = {"test_prevalence": 1.0, "full_dataset_prevalence": (fraud_n * (1 - pi) / pi) / legit_n}
    models = {"production": ctx.production_proba()}
    for name in ("B", "C2"):
        if ctx.proba_path(name).is_file():
            models[f"ablation_{name}"] = np.load(ctx.proba_path(name))
    default_costs = {"missed_fraud_cost": "the transaction amount of every missed fraud (x multiplier)",
                     "false_hold_cost": 10.0, "analyst_review_cost": 2.0}
    sensitivities = [(10.0, 2.0), (2.0, 0.5), (50.0, 2.0), (10.0, 10.0), (50.0, 10.0)]
    out: dict = {
        "HYPOTHETICAL_ASSUMPTIONS": {
            "warning": "No real cost data exists. All costs below are hypothetical and exist only to compare policies.",
            "missed_fraud": "loss = transaction amount of each fraud that is neither reviewed nor held",
            "false_hold": "flat customer-friction cost per LEGITIMATE transaction that is held (score >= hold threshold)",
            "analyst_review": "flat cost per reviewed alert (score >= review threshold, fraud or legitimate)",
            "analyst_assumption": "an analyst correctly identifies every fraud among reviewed alerts (optimistic)",
            "defaults": default_costs, "sensitivity_grid_(false_hold, review)": sensitivities,
            "current_policy": {"review_threshold": 0.60, "hold_threshold": 0.85},
        },
        "grid": grid, "models": {},
    }
    for model_name, proba in models.items():
        out["models"][model_name] = {}
        for weight_name, weight in weights.items():
            entries = []
            for (hold_cost, review_cost) in sensitivities:
                best = None
                for t_review in grid:
                    for t_hold in grid:
                        if t_hold < t_review:
                            continue
                        cost = _policy_cost(yt, amount, proba, t_review, t_hold, 1.0, hold_cost, review_cost, weight)
                        if best is None or cost["total_cost"] < best[2]["total_cost"]:
                            best = (t_review, t_hold, cost)
                current = _policy_cost(yt, amount, proba, 0.60, 0.85, 1.0, hold_cost, review_cost, weight)
                reference = _policy_cost(yt, amount, proba, 0.30, 0.85, 1.0, hold_cost, review_cost, weight)
                entries.append({
                    "false_hold_cost": hold_cost, "analyst_review_cost": review_cost,
                    "current_thresholds": {"review": 0.60, "hold": 0.85, **current},
                    "alternative_review_0.30_hold_0.85": reference,
                    "best_on_grid": {"review": best[0], "hold": best[1], **best[2]},
                    "current_cost_over_best": (current["total_cost"] / best[2]["total_cost"]) if best[2]["total_cost"] else None,
                })
            out["models"][model_name][weight_name] = entries
    # Does the choice of threshold matter for the production model at all?
    prod = models["production"]
    out["production_score_separation"] = {
        "rows_in_review_band_0.30_to_0.85": int(((prod >= 0.30) & (prod < 0.85)).sum()),
        "decisions_change_between_thresholds_0.05_and_0.95": int(((prod >= 0.05) != (prod >= 0.95)).sum()),
    }
    c.save_json("thresholds", out)
    e = out["models"]["production"]["full_dataset_prevalence"][0]
    print("[thresholds] production, default costs: current/best =", e["current_cost_over_best"], flush=True)


# ---------------------------------------------------------------- fairness
def section_fairness(ctx: Context) -> None:
    ctx.load()
    test_df = ctx.df[ctx.test].reset_index(drop=True)
    yt = ctx.y[ctx.test]
    train_amounts = ctx.df.loc[ctx.train, "amount"]
    edges = [float(train_amounts.quantile(q)) for q in (0.50, 0.90, 0.99)]
    bands = np.select([test_df["amount"] <= edges[0], test_df["amount"] <= edges[1], test_df["amount"] <= edges[2]],
                      ["low (<= train P50)", "medium (P50-P90)", "high (P90-P99)"], "very high (> train P99)")
    models = {"production": ctx.production_proba()}
    if ctx.proba_path("C2").is_file():
        models["ablation_C2"] = np.load(ctx.proba_path("C2"))
    out = {
        "scope": ("PaySim has no demographic, KYC, gender, age, geography or account-holder attributes, so "
                  "demographic fairness CANNOT be evaluated. Only transaction-type and amount-band slices are possible."),
        "amount_band_edges_from_train_split": {"P50": edges[0], "P90": edges[1], "P99": edges[2]},
        "caveat": "Slices with few fraud rows give unstable rates; counts are shown for that reason.",
        "models": {},
    }
    for model_name, proba in models.items():
        pred = proba >= 0.5
        slices = {}
        for dimension, labels in (("transaction_type", test_df["type"].to_numpy()), ("amount_band", bands)):
            slices[dimension] = {}
            for label in sorted(set(labels)):
                m = labels == label
                y_s, p_s = yt[m], pred[m]
                legit, fraud = int((y_s == 0).sum()), int((y_s == 1).sum())
                fp, fn = int((p_s & (y_s == 0)).sum()), int((~p_s & (y_s == 1)).sum())
                tp = fraud - fn
                slices[dimension][str(label)] = {
                    "n": int(m.sum()), "fraud": fraud, "legit": legit, "false_positives": fp, "false_negatives": fn,
                    "false_positive_rate": fp / legit if legit else None,
                    "false_negative_rate": fn / fraud if fraud else None,
                    "precision": tp / (tp + fp) if (tp + fp) else None,
                    "recall": tp / fraud if fraud else None,
                }
        out["models"][model_name] = slices
    c.save_json("fairness", out)
    print("[fairness] done", flush=True)


# ---------------------------------------------------------------- adversarial
def _score_rows(booster, rows: pd.DataFrame, strict: bool = True) -> np.ndarray:
    rows = rows.sort_values("step", kind="stable").reset_index(drop=True)
    features = c.build_features(rows, strict_same_step=strict)
    return c.predict_production(booster, features)


def section_adversarial(ctx: Context) -> None:
    ctx.load()
    booster, _ = c.load_production_models()
    rng = np.random.default_rng(42)
    test_df = ctx.df[ctx.test].reset_index(drop=True)
    fraud = test_df[(test_df["isFraud"] == 1) & test_df["type"].isin(["TRANSFER", "CASH_OUT"])]
    legit = test_df[(test_df["isFraud"] == 0) & test_df["type"].isin(["TRANSFER", "CASH_OUT"])]
    n = min(300, len(fraud))
    fraud = fraud.sample(n=n, random_state=42).reset_index(drop=True)
    legit = legit.sample(n=min(300, len(legit)), random_state=42).reset_index(drop=True)
    scenarios: list[dict] = []

    def summarize(name, modification, scores_by_case, observation, limitation, controls=None):
        detect = float(np.mean([s.max() >= 0.5 for s in scores_by_case]))
        every = float(np.mean([bool((s >= 0.5).all()) for s in scores_by_case]))
        flat = np.concatenate(scores_by_case)
        entry = {"scenario": name, "input_modification": modification, "cases": len(scores_by_case),
                 "share_of_cases_with_at_least_one_part_scored_>=0.5": detect,
                 "share_of_cases_with_all_parts_scored_>=0.5": every,
                 "share_of_scored_rows_>=0.5": float((flat >= 0.5).mean()),
                 "mean_score": float(flat.mean()), "median_score": float(np.median(flat)),
                 "observation": observation, "limitation": limitation}
        if controls is not None:
            entry["legit_controls_share_scored_>=0.5"] = controls
        scenarios.append(entry)

    # 0. Unmodified fraud rows.
    base = [np.array([s]) for s in _score_rows(booster, fraud)]
    base_detect = float(np.mean([s[0] >= 0.5 for s in base]))
    legit_base = float((_score_rows(booster, legit) >= 0.5).mean())
    summarize("unmodified fraud rows", "none", base,
              f"{base_detect:.1%} of sampled fraud rows score >= 0.5; {legit_base:.1%} of sampled legitimate "
              "TRANSFER/CASH_OUT rows score >= 0.5.", "Reference point only.", controls=legit_base)

    def split_case(row, k, spread_steps):
        a = row["amount"] / k
        parts = []
        for i in range(k):
            old_o = row["oldbalanceOrg"] - i * a
            parts.append({
                "step": int(row["step"]) + (i if spread_steps else 0), "type": row["type"], "amount": a,
                "nameOrig": "SPLITSENDER", "oldbalanceOrg": old_o, "newbalanceOrig": max(old_o - a, 0.0),
                "nameDest": row["nameDest"], "oldbalanceDest": row["oldbalanceDest"] + i * a,
                "newbalanceDest": row["oldbalanceDest"] + (i + 1) * a, "isFraud": 1,
            })
        return pd.DataFrame(parts)

    for k in (2, 4, 10):
        cases = [_score_rows(booster, split_case(row, k, False), strict=True) for _, row in fraud.iterrows()]
        summarize(f"amount splitting x{k} (same step)",
                  f"split each fraud amount into {k} equal consecutive transfers from the same account, consistent balances, same step",
                  cases, "Only the final part empties the account; earlier parts leave a residual balance.",
                  "Same-step history is ignored by the strict point-in-time policy, so history features cannot help here.")
    for k in (4,):
        cases = [_score_rows(booster, split_case(row, k, True), strict=True) for _, row in fraud.iterrows()]
        summarize(f"gradual / repeated smaller transfers x{k} (one per step)",
                  f"same as splitting but one part per consecutive step so sender history exists",
                  cases, "History features exist for later parts, but they were inert in training (SHAP 0).",
                  "The model has no velocity feature; the user_* history features were effectively constant in training.")

    # Recipient change: model has no identifier features; only destination balances can matter.
    def recipient_zero(r):
        r = r.copy()
        r["oldbalanceDest"], r["newbalanceDest"], r["nameDest"] = 0.0, 0.0, "NEWDEST"
        return r

    def recipient_credited(r):
        r = r.copy()
        r["newbalanceDest"], r["nameDest"] = r["oldbalanceDest"] + r["amount"], "NEWDEST"
        return r

    for label, mutate in (
        ("recipient change: new recipient account (zero balances)", recipient_zero),
        ("recipient change: recipient credited consistently", recipient_credited),
    ):
        rows = pd.DataFrame([mutate(r) for _, r in fraud.iterrows()])
        scores = _score_rows(booster, rows)
        summarize(label, "change nameDest (no model feature) and/or destination balances",
                  [np.array([s]) for s in scores],
                  "The model has no recipient-identity feature; only the destination balance features respond.",
                  "Recipient identity is invisible to the XGBoost model; only network signals could see it.")

    # Transaction type change.
    for new_type in ("TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"):
        rows = fraud.copy()
        rows["type"] = new_type
        scores = _score_rows(booster, rows)
        lrows = legit.copy()
        lrows["type"] = new_type
        summarize(f"type change -> {new_type}", f"set type={new_type}, all other fields unchanged",
                  [np.array([s]) for s in scores],
                  "Changes the five one-hot type features only.",
                  "PaySim fraud only occurs as TRANSFER/CASH_OUT, so other types are out-of-distribution for fraud.",
                  controls=float((_score_rows(booster, lrows) >= 0.5).mean()))

    # Balance-consistency manipulation (probes the known shortcut).
    for label, new_balance in (("sender new balance +1 (account not fully emptied)", lambda r: 1.0),
                               ("sender new balance 1% of old balance", lambda r: r["oldbalanceOrg"] * 0.01),
                               ("sender new balance 10% of old balance", lambda r: r["oldbalanceOrg"] * 0.10)):
        rows = fraud.copy()
        rows["newbalanceOrig"] = [new_balance(r) for _, r in fraud.iterrows()]
        lrows = legit.copy()
        scores = _score_rows(booster, rows)
        summarize(f"balance manipulation: {label}", "break the exact balance accounting (orig_balance_error != 0)",
                  [np.array([s]) for s in scores],
                  "Shows how strongly the model relies on exact balance consistency (known PaySim shortcut).",
                  "This is an input the sender cannot freely control in a real system; it is a diagnostic, not a threat model.",
                  controls=float((_score_rows(booster, lrows) >= 0.5).mean()))

    c.save_json("adversarial", {
        "note": ("Controlled stress tests with the PRODUCTION model on PaySim strict-test rows. They describe model behaviour "
                 "under input changes. They do NOT demonstrate robustness; every row that scores >= 0.5 after a change "
                 "is simply an observation."),
        "sample": {"fraud_rows": int(len(fraud)), "legit_control_rows": int(len(legit)), "seed": 42,
                   "source": "strict test split, TRANSFER/CASH_OUT rows"},
        "scenarios": scenarios,
    })
    print("[adversarial]", len(scenarios), "scenarios", flush=True)


# ---------------------------------------------------------------- signals
def section_signals(ctx: Context) -> None:
    ctx.load()
    df = ctx.df
    test_idx = np.flatnonzero(ctx.test)
    t = df.iloc[test_idx]
    y = ctx.y[test_idx]
    dest_code, _ = pd.factorize(df["nameDest"])
    orig_code, _ = pd.factorize(df["nameOrig"])
    pair_code, _ = pd.factorize(df["nameOrig"].astype(str) + "|" + df["nameDest"].astype(str))
    step = df["step"].to_numpy().astype(np.int64)

    def prior_unique(group_codes, other_codes):
        """Number of distinct `other` partners seen with each group at steps strictly before the row's step."""
        first = pd.DataFrame({"g": group_codes, "o": other_codes, "s": step}).groupby(["g", "o"], sort=False)["s"].min().reset_index()
        key = np.sort(first["g"].to_numpy().astype(np.int64) * 10_000 + first["s"].to_numpy())
        q = group_codes[test_idx].astype(np.int64) * 10_000
        return np.searchsorted(key, q + step[test_idx], "left") - np.searchsorted(key, q, "left")

    def prior_pair_count(codes):
        key = np.sort(codes.astype(np.int64) * 10_000 + step)
        q = codes[test_idx].astype(np.int64) * 10_000
        return np.searchsorted(key, q + step[test_idx], "left") - np.searchsorted(key, q, "left")

    prior_senders = prior_unique(dest_code, orig_code)
    prior_recipients = prior_unique(orig_code, dest_code)
    prior_pair = prior_pair_count(pair_code)
    new_relationship = prior_pair == 0
    # Production counts include the current transaction (a new relationship adds one).
    senders_incl = prior_senders + new_relationship
    recipients_incl = prior_recipients + new_relationship
    fires_recipient = senders_incl >= 5
    fires_sender = recipients_incl >= 5
    tc_mask = t["type"].isin(["TRANSFER", "CASH_OUT"]).to_numpy()

    def summary(mask):
        out = {}
        for label, m in (("fraud", mask & (y == 1)), ("legit", mask & (y == 0))):
            if not m.any():
                out[label] = None
                continue
            out[label] = {
                "n": int(m.sum()),
                "recipient_unique_senders_incl_current": {"mean": float(senders_incl[m].mean()), "median": float(np.median(senders_incl[m])),
                                                         "p95": float(np.quantile(senders_incl[m], 0.95))},
                "sender_unique_recipients_incl_current": {"mean": float(recipients_incl[m].mean()), "max": int(recipients_incl[m].max())},
                "new_relationship_share": float(new_relationship[m].mean()),
                "HIGH_RECIPIENT_CONNECTIVITY_fires": float(fires_recipient[m].mean()),
                "HIGH_SENDER_CONNECTIVITY_fires": float(fires_sender[m].mean()),
            }
        return out

    c.save_json("signals", {
        "note": ("Point-in-time (strictly earlier steps) re-computation of the production network signals over the whole PaySim "
                 "history for the strict test rows, compared between fraud and legitimate rows. Descriptive only; these signals "
                 "do not feed the model or the risk score."),
        "thresholds": {"unique_senders": 5, "unique_recipients": 5, "counts_include_current_transaction": True},
        "all_test_rows": summary(np.ones(len(y), dtype=bool)),
        "transfer_and_cash_out_only": summary(tc_mask),
        "sender_history_availability": {
            "test_rows_with_any_prior_transaction_by_same_sender": int((ctx.X.loc[ctx.test, "user_tx_count_before"] > 0).sum()),
            "test_rows": int(len(y)),
            "meaning": "Sender velocity and amount-vs-history signals are almost never available on PaySim.",
        },
        "behavioral_note": "Because nameOrig almost never repeats, sender-velocity behavioural signals cannot be evaluated on PaySim.",
    })
    print("[signals] done", flush=True)


# ---------------------------------------------------------------- report
def _fmt(value, digits=6):
    return "n/a" if value is None else f"{value:.{digits}f}"


def section_report(_: Context) -> None:
    lines = ["# TrustX evaluation report", "",
             "> All numbers below are generated by `evaluation/run_evaluation.py` on the PaySim **synthetic** dataset "
             "(strict temporal split). They are not real-world or real upay performance.", ""]
    bench = c.load_json("benchmark")
    if bench:
        r = bench["production_model_strict_test"]
        cm = r["confusion_matrix"]
        s = bench["split"]
        lines += ["## 1. Production model benchmark (strict temporal test set)", "",
                  f"Reproduced from `Models/final_xgb_model.json`: matches README benchmark = **{bench['all_match']}**", "",
                  "| Metric | Value |", "|---|---|",
                  f"| Precision | {_fmt(r['precision'])} |", f"| Recall | {_fmt(r['recall'])} |", f"| F1 | {_fmt(r['f1'])} |",
                  f"| ROC-AUC | {_fmt(r['roc_auc'])} |", f"| PR-AUC | {_fmt(r['pr_auc'])} |",
                  f"| Confusion matrix | TN {cm['tn']:,} / FP {cm['fp']:,} / FN {cm['fn']:,} / TP {cm['tp']:,} |", "",
                  "| Split | Rows | Fraud | Prevalence | Steps | Fraud/step | Legit/step |", "|---|---|---|---|---|---|---|"]
        for name in ("train", "val", "test"):
            x = s[name]
            lines.append(f"| {name} | {x['rows']:,} | {x['fraud']:,} | {x['prevalence']:.3%} | {x['steps']} | {x['fraud_per_step']:.1f} | {x['legit_per_step']:.0f} |")
        sc = bench["shortcut_check"]
        lines += ["", f"Dataset prevalence: {s['dataset_prevalence']:.3%}; strict test-set prevalence: {s['test']['prevalence']:.3%}. "
                  f"Fraud per step is {s['train']['fraud_per_step']:.1f} / {s['val']['fraud_per_step']:.1f} / {s['test']['fraud_per_step']:.1f} "
                  f"(train / val / test) while legitimate rows per step are {s['train']['legit_per_step']:.0f} / {s['val']['legit_per_step']:.0f} / "
                  f"{s['test']['legit_per_step']:.0f}. Precision, alert volume and false-positive counts measured on this test set therefore "
                  "do not transfer to a lower-prevalence population.", "",
                  f"`orig_balance_error == 0`: {sc['orig_balance_error_is_zero_share_fraud']:.1%} of fraud rows vs "
                  f"{sc['orig_balance_error_is_zero_share_legit']:.1%} of legitimate rows (test set).", "",
                  f"Strict point-in-time `user_*` features: {bench['strict_pit_history_variant']['rows_whose_user_features_differ_from_notebook_policy']:,} rows differ from the notebook policy; "
                  f"strict-policy F1 = {_fmt(bench['strict_pit_history_variant']['metrics']['f1'])}.", ""]
    rows = []
    for name in ("A_retrain", "B", "C", "C2", "D", "E", "F", "G"):
        a = c.load_json(f"ablation_{name}")
        if a:
            cm = a["confusion_matrix"]
            rows.append(f"| {name} | {a['description']} | {a['features_used']} | {_fmt(a['precision'],4)} | {_fmt(a['recall'],4)} | {_fmt(a['f1'],4)} | {_fmt(a['roc_auc'],5)} | {_fmt(a['pr_auc'],4)} | {cm['tn']:,}/{cm['fp']:,}/{cm['fn']:,}/{cm['tp']:,} |")
    if rows:
        lines += ["## 2. Ablations (retrained in memory, same hyper-parameters, CPU)", "",
                  "| ID | Variant | Features | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN/FP/FN/TP |", "|---|---|---|---|---|---|---|---|---|", *rows, "",
                  "Test prevalence is 1.4% for every row (a model with no skill has PR-AUC about 0.014).", ""]
        e = c.load_json("ablation_E")
        if e and e.get("note"):
            lines += ["**Variant E:** " + e["note"], ""]
    base = c.load_json("baselines")
    if base:
        lines += ["## 3. Baselines (same strict test set)", "", "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN/FP/FN/TP |", "|---|---|---|---|---|---|---|"]
        for name, r in base["models"].items():
            cm = r["confusion_matrix"]
            lines.append(f"| {name} | {_fmt(r['precision'],4)} | {_fmt(r['recall'],4)} | {_fmt(r['f1'],4)} | {_fmt(r['roc_auc'],4)} | {_fmt(r['pr_auc'],4)} | {cm['tn']:,}/{cm['fp']:,}/{cm['fn']:,}/{cm['tp']:,} |")
        lines.append("")
    iso = c.load_json("isolation_forest")
    if iso:
        q = iso["ranking_quality_of_negative_decision_function"]
        n = iso["native_predict_minus_one_as_flag"]
        lines += ["## 4. Isolation Forest (anomaly signal, not part of the risk score)", "",
                  f"ROC-AUC {_fmt(q['roc_auc'],4)}, PR-AUC {_fmt(q['pr_auc'],4)} (prevalence {q['prevalence']:.3%}). "
                  f"Native `predict() == -1` flag: precision {_fmt(n['precision'],4)}, recall {_fmt(n['recall'],4)}. "
                  f"Mean decision value: fraud {iso['decision_function_mean']['fraud']:.4f}, legit {iso['decision_function_mean']['legit']:.4f}. "
                  f"Share below 0: fraud {iso['share_below_zero']['fraud']:.1%}, legit {iso['share_below_zero']['legit']:.1%}.", ""]
    risk = c.load_json("risk")
    if risk:
        lines += ["## 5. Risk engine and alert volume", ""]
        for name, t in risk.items():
            if not isinstance(t, dict) or "levels" not in t:
                continue
            lv = t["levels"]
            lines += [f"**{name}**", "", "| Level | Count | Fraud | Fraud rate |", "|---|---|---|---|"]
            for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
                x = lv[level]
                lines.append(f"| {level} | {x['count']:,} | {x['fraud']:,} | {_fmt(x['fraud_rate'],4)} |")
            lines += ["", f"Review queue (HIGH+CRITICAL): {t['review_queue_size']:,} rows, precision {_fmt(t['review_queue_precision'],4)}, "
                      f"recall {_fmt(t['review_queue_recall'],4)}. Alerts per 100,000 transactions: "
                      f"{t['alerts_per_100k_at_test_prevalence']:.0f} at test prevalence; "
                      f"{t['alerts_per_100k_reweighted_to_full_dataset_prevalence']:.0f} re-weighted to the full-dataset prevalence. "
                      f"Rows with 0.05 < p < 0.95: fraud {t['rows_with_probability_between_0.05_and_0.95']['fraud']}, legit {t['rows_with_probability_between_0.05_and_0.95']['legit']}.", ""]
    thr = c.load_json("thresholds")
    if thr:
        lines += ["## 6. Threshold cost analysis (HYPOTHETICAL costs)", "",
                  "> " + thr["HYPOTHETICAL_ASSUMPTIONS"]["warning"], "",
                  "| Model | Prevalence weighting | False-hold / review cost | Current (0.60/0.85) cost per 100k | Best on grid | Best cost per 100k | Current / best |",
                  "|---|---|---|---|---|---|---|"]
        for model_name, by_weight in thr["models"].items():
            for weight_name, entries in by_weight.items():
                for e in entries:
                    b = e["best_on_grid"]
                    lines.append(f"| {model_name} | {weight_name} | {e['false_hold_cost']:g} / {e['analyst_review_cost']:g} | {e['current_thresholds']['cost_per_100k']:.1f} | "
                                 f"review {b['review']:g} / hold {b['hold']:g} | {b['cost_per_100k']:.1f} | {_fmt(e['current_cost_over_best'],2)} |")
        sep = thr.get("production_score_separation", {})
        lines += ["", "Ties are broken towards the first grid point, so the reported 'best' policy is the lowest-threshold one among equal-cost policies. "
                  f"For the production model, {sep.get('rows_in_review_band_0.30_to_0.85')} test rows fall between 0.30 and 0.85 and only "
                  f"{sep.get('decisions_change_between_thresholds_0.05_and_0.95')} rows change decision between a 0.05 and a 0.95 threshold: its scores are "
                  "almost binary, so the threshold choice is immaterial for it on this data. Missed-fraud cost is the transaction amount, which is "
                  "large compared with the flat hypothetical review and hold costs, so cheaper-to-review policies win whenever a model is less separable.", ""]
    fair = c.load_json("fairness")
    if fair:
        lines += ["## 7. Slice analysis", "", "> " + fair["scope"], ""]
        for model_name, dims in fair["models"].items():
            for dimension, slices in dims.items():
                lines += [f"**{model_name} by {dimension}**", "", "| Slice | n | Fraud | FP | FN | FPR | FNR |", "|---|---|---|---|---|---|---|"]
                for label, x in slices.items():
                    lines.append(f"| {label} | {x['n']:,} | {x['fraud']:,} | {x['false_positives']:,} | {x['false_negatives']:,} | {_fmt(x['false_positive_rate'],5)} | {_fmt(x['false_negative_rate'],4)} |")
                lines.append("")
    adv = c.load_json("adversarial")
    if adv:
        lines += ["## 8. Adversarial stress tests", "", "> " + adv["note"], "",
                  "| Scenario | Cases | >=1 part flagged | All parts flagged | Rows >= 0.5 | Mean score | Legit control >= 0.5 |", "|---|---|---|---|---|---|---|"]
        for s in adv["scenarios"]:
            lines.append(f"| {s['scenario']} | {s['cases']} | {s['share_of_cases_with_at_least_one_part_scored_>=0.5']:.1%} | "
                         f"{s['share_of_cases_with_all_parts_scored_>=0.5']:.1%} | {s['share_of_scored_rows_>=0.5']:.1%} | {s['mean_score']:.4f} | "
                         f"{_fmt(s.get('legit_controls_share_scored_>=0.5'),3)} |")
        lines.append("")
    sig = c.load_json("signals")
    if sig:
        lines += ["## 9. Network signals, fraud vs legitimate (strictly earlier steps)", "", "> " + sig["note"], ""]
        for scope in ("all_test_rows", "transfer_and_cash_out_only"):
            lines += [f"**{scope}**", "", "| Class | n | Recipient unique senders (mean / median) | New relationship | HIGH_RECIPIENT_CONNECTIVITY fires | HIGH_SENDER_CONNECTIVITY fires |", "|---|---|---|---|---|---|"]
            for label in ("fraud", "legit"):
                x = sig[scope][label]
                if x:
                    r = x["recipient_unique_senders_incl_current"]
                    lines.append(f"| {label} | {x['n']:,} | {r['mean']:.2f} / {r['median']:.0f} | {x['new_relationship_share']:.1%} | {x['HIGH_RECIPIENT_CONNECTIVITY_fires']:.1%} | {x['HIGH_SENDER_CONNECTIVITY_fires']:.1%} |")
            lines.append("")
        av = sig["sender_history_availability"]
        lines.append(f"Sender history exists for {av['test_rows_with_any_prior_transaction_by_same_sender']} of {av['test_rows']:,} test rows. {sig['behavioral_note']}")
    (c.RESULTS_DIR / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[report] written", c.RESULTS_DIR / "evaluation_report.md", flush=True)


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", default="all", help="comma-separated: " + ",".join(ALL_SECTIONS) + " or 'all'")
    parser.add_argument("--variants", default=",".join(ABLATIONS), help="ablation variants to (re)train")
    parser.add_argument("--csv", default=None, help="PaySim CSV path (or env PAYSIM_CSV)")
    parser.add_argument("--cache-dir", default=str(c.DEFAULT_CACHE), help="scratch dir for cached score arrays (not in repo)")
    parser.add_argument("--quick", action="store_true", help="smoke-test on the first rows only; results go to results/quick/")
    parser.add_argument("--quick-rows", type=int, default=400_000)
    args = parser.parse_args()
    if args.quick:
        c.RESULTS_DIR = c.RESULTS_DIR / "quick"
        c.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sections = ALL_SECTIONS if args.sections == "all" else [s.strip() for s in args.sections.split(",")]
    before = c.models_fingerprint()
    ctx = Context(args)
    handlers = {
        "benchmark": section_benchmark, "baselines": section_baselines,
        "isolation_forest": section_isolation_forest, "risk": section_risk,
        "thresholds": section_thresholds, "fairness": section_fairness,
        "adversarial": section_adversarial, "signals": section_signals, "report": section_report,
    }
    for section in sections:
        t0 = time.time()
        if section == "ablation":
            section_ablation(ctx, [v.strip() for v in args.variants.split(",") if v.strip()])
        else:
            handlers[section](ctx)
        print(f"[{section}] finished in {time.time() - t0:.0f}s", flush=True)
    c.guard_models_unchanged(before)
    print("Models/ artifacts unchanged.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
