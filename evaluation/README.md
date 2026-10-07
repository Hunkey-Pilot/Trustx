# TrustX evaluation suite

Reproducible, read-only evaluation of the **existing** TrustX models on the **PaySim synthetic dataset**, using the
production model's strict temporal split (train steps ≤ 520, validation 521–631, test ≥ 632).

* Production artifacts in `Models/` are only **loaded**. The script records their modification times before the run and
  aborts if any changed (`Models/ artifacts unchanged.` is printed at the end).
* Ablation and baseline models are trained **in memory** with the production hyper-parameters (CPU) and never saved.
* Nothing here measures real-world or real upay performance.

## Run

```powershell
cd D:\TrustX
# needs the PaySim CSV: --csv <path> or env PAYSIM_CSV (default: the author's Downloads path)
.\backend\.venv\Scripts\python.exe evaluation\run_evaluation.py --sections all
.\backend\.venv\Scripts\python.exe evaluation\run_evaluation.py --sections risk,thresholds,report
.\backend\.venv\Scripts\python.exe evaluation\run_evaluation.py --sections ablation --variants B,C2
.\backend\.venv\Scripts\python.exe evaluation\run_evaluation.py --quick        # 400k-row smoke test
```

Sections: `benchmark`, `ablation`, `baselines`, `isolation_forest`, `risk`, `thresholds`, `fairness`, `adversarial`,
`signals`, `report`. A full run takes roughly an hour on CPU (each ablation retrains XGBoost on 6.08M rows).
Intermediate score arrays are cached outside the repo (`%TEMP%\trustx_eval_cache`). Latency profiling lives in
`scripts/profile_latency.py`.

## Outputs (`evaluation/results/`)

| File | Content |
|---|---|
| `evaluation_report.md` | Everything below in one readable report (generated, numbers copied from the JSON files) |
| `benchmark.json` | Production model on the strict test set; check against the README benchmark; split statistics; strict point-in-time variant |
| `ablation_<ID>.json` | One file per ablation (A_retrain, B, C, C2, D, E, F, G) |
| `baselines.json` | Logistic regression, random forest, rule baseline, amount-threshold baseline |
| `isolation_forest.json` | ROC-AUC / PR-AUC and score distributions of the Isolation Forest |
| `risk.json` | Risk-level distribution, fraud rate per level, queue sizes and precision, alerts per 100,000 transactions |
| `thresholds.json` | Cost-based threshold analysis (**hypothetical costs**) |
| `fairness.json` | False-positive / false-negative rates by transaction type and amount band |
| `adversarial.json` | Stress tests: splitting, gradual transfers, recipient change, type change, balance manipulation |
| `signals.json` | Network connectivity signals, fraud vs legitimate, strictly earlier steps |
| `latency_profile*.json` | Local latency profile from `scripts/profile_latency.py` |

## Method notes

* **Features:** a vectorised re-implementation of `backend/app/services/feature_engineering.py`. The default
  (`user_*` features via `cumcount`, as in the training notebook) reproduces the shipped model's benchmark exactly. A
  `strict_same_step=True` variant (runtime policy: only strictly earlier steps) is also evaluated; it changes the `user_*`
  features of 4,776 test rows and leaves the metrics unchanged.
* **Ablation IDs:** A_retrain = all 28 features retrained on CPU (control; reproduces the artifact's confusion matrix);
  B = no `orig_balance_error`; C = no `orig_balance_error` and no `dest_balance_error`; C2 = none of the six balance-derived
  features; D = transaction type only; E = the five `user_*` features only; F = no time features; G = neither balance-derived
  nor time features. *All ablations remove derived columns only; the raw balance columns remain in every variant except D and E.*
* **Baselines:** logistic regression and random forest are trained on all training fraud rows plus a random sample of
  legitimate rows (1,000,000 and 400,000 respectively, seed 42); the amount threshold is chosen on the training split only.
* **Risk analysis:** counts are computed from the production model's probabilities with the production thresholds
  (30/60/85 on probability × 100). "Re-weighted" alert volume gives legitimate rows the weight that makes fraud
  prevalence equal to the full-dataset 0.129%. With 0 observed false positives the 95% upper bound on the false-positive
  rate is 3/n_legit (rule of three), not zero.
* **Threshold analysis (HYPOTHETICAL):** no real cost data exists. Missed fraud costs the transaction amount; a held legitimate
  transaction costs a flat friction value; every reviewed alert costs a flat analyst value; analysts are assumed to catch
  every fraud they review. A small sensitivity grid is reported. Ties favour the lowest-threshold policy.
* **Adversarial tests** feed modified PaySim strict-test fraud rows (300 sampled, seed 42) through the same feature builder
  and the production model. They are observations about the model, not a robustness claim.

## Key results (strict test set, PaySim synthetic, 89,466 rows, 1,252 fraud = 1.399%)

Production model reproduced: precision 1.000000, recall 0.999201, F1 0.999600, ROC-AUC 1.000000, PR-AUC 0.999994,
TN 88,214 / FP 0 / FN 1 / TP 1,251 (`benchmark.json`).

| Model / variant | Precision | Recall | F1 | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|
| A: 28 features, retrained (CPU) | 1.0000 | 0.9992 | 0.9996 | 1.0000 | 0 | 1 |
| B: no `orig_balance_error` | 0.9505 | 0.9976 | 0.9735 | 0.9961 | 65 | 3 |
| C: no balance-error features | 0.9506 | 0.9992 | 0.9743 | 0.9959 | 65 | 1 |
| C2: no balance-derived features | 0.6441 | 0.9976 | 0.7828 | 0.9907 | 690 | 3 |
| D: transaction type only | 0.0330 | 1.0000 | 0.0638 | 0.0519 | 36,727 | 0 |
| E: `user_*` history features only | 0.0532 | 0.5567 | 0.0971 | 0.1702 | 12,403 | 555 |
| F: no time features | 0.9992 | 0.9992 | 0.9992 | 1.0000 | 1 | 1 |
| G: no balance-derived, no time | 0.7964 | 0.9968 | 0.8854 | 0.9894 | 319 | 4 |
| Logistic regression | 0.1968 | 0.9577 | 0.3264 | 0.8058 | 4,895 | 53 |
| Random forest | 1.0000 | 0.9992 | 0.9996 | 1.0000 | 0 | 1 |
| Rule: TRANSFER/CASH_OUT and account emptied | 0.0594 | 0.9585 | 0.1119 | 0.0575 | 19,004 | 52 |
| Amount threshold (chosen on train) | 0.4817 | 0.1470 | 0.2252 | 0.0827 | 198 | 1,068 |

What this shows, and what it does not:

* A random forest reaches **the same confusion matrix** as the XGBoost model. PaySim's fraud is very separable for tree
  models once balance features are available; the XGBoost result is not evidence of an unusually capable model.
* Removing `orig_balance_error` costs ~2.6 F1 points and 65 false positives; removing all six balance-derived features costs ~22
  F1 points and 690 false positives. Because the **raw balance columns remain** in these variants, the ablations do not show
  how the model would behave without any balance information.
* Variant E is not evidence for "behavioral" value: 99.7% of test rows have no sender history, so `amount_vs_user_avg`
  equals the raw amount and the signal is just the amount.
* Test-set precision and alert volumes depend on the 1.4% prevalence; at the full-dataset prevalence the production review queue is
  about 129 alerts per 100,000 transactions (all true fraud, because no false positive was observed), not 1,398.
* The production model's scores are almost binary: only 1 of 89,466 test rows (legitimate) has a score between 0.05 and 0.95, so
  MEDIUM and HIGH are empty and the threshold choice is immaterial for it. For less separable models (B, C2, G) the
  hypothetical-cost analysis favours low review thresholds, because a missed fraud costs its full amount.
* Isolation Forest: ROC-AUC 0.8853, PR-AUC 0.3826; 50.7% of fraud and 1.3% of legitimate rows have a negative decision value.
* Network connectivity patterns fire on **more** legitimate than fraudulent rows on PaySim (HIGH_RECIPIENT_CONNECTIVITY: 20.9% of fraud
  vs 37.2% of legitimate rows overall; 55.5% of legitimate TRANSFER/CASH_OUT rows). They are descriptive context, not a fraud detector.
* Splitting an amount into 2/4/10 transfers drops the production model's detection of at least one part to 96.7% / 65.0% / 52.0%
  and the share of flagged parts to 49.5% / 16.2% / 5.2%; the model has no velocity or aggregation feature. Making the sender's new balance
  non-zero cuts detection to 22–27%. **The model is not robust to these changes.** It ignores the recipient's identity entirely.
* Fairness: demographic/KYC fairness **cannot** be evaluated with PaySim. Slice results by transaction type and amount band are in
  `fairness.json`; the production model has zero false positives in every slice, so the slices are uninformative for it, and the
  C2 ablation's false-positive rate varies from 0.00% to 0.99% across amount bands.
