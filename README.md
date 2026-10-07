# TrustX

### AI Trust & Risk Intelligence Platform

TrustX is a **hackathon prototype** that helps an analyst review potentially suspicious mobile-money transactions.
It combines an XGBoost fraud model, an Isolation Forest anomaly signal, SHAP explanations, behavioral and network
signals, hypothetical counterfactual scenarios, a deterministic investigator summary, and an analyst review workflow,
on FastAPI, PostgreSQL and Next.js.

> **Hackathon:** AI Hackathon 2026 — DIU CPC × upay · **Track:** Trust & Risk
>
> **Read this first.** The models were trained and evaluated only on **PaySim, a synthetic simulator**. Every metric in
> this repository is a *PaySim benchmark result*. None of it is real-world, real upay, or production performance, and
> the "fraud probability" is **not** a calibrated real-world probability. See [Limitations](#limitations).

---

## Problem and solution

Mobile financial services (MFS) such as upay move money at very high volume; analysts cannot read every transaction.
TrustX scores each submitted transaction, explains the score, adds simple relationship and history signals, routes
HIGH and CRITICAL cases to a human review queue, and records the analyst's decision separately from the model output.

What TrustX deliberately does **not** do: it never declares a transaction "confirmed fraud", never labels an account a
mule, and never acts automatically. HIGH and CRITICAL assessments require human review.

Which real MFS scenarios (SIM swap, social engineering, agent abuse, fake reversal, mule-like behavior, unusual
transfers) the current data can and cannot address is documented in [`docs/MFS_SCENARIOS.md`](docs/MFS_SCENARIOS.md).

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the diagram of what is actually implemented.

```text
Browser (Next.js) -> same-origin proxy route (adds server-side API key)
                  -> FastAPI (API-key auth: ANALYST / ADMIN)
                       -> feature engineering (28 features) -> XGBoost, Isolation Forest, SHAP
                       -> risk engine -> evidence engine -> investigator summary
                       -> behavioral + network signals (earlier steps only, persisted)
                       -> PostgreSQL: transaction_analyses, case_reviews, audit_logs
```

Not part of the system: Redis, Kafka/Celery, WebSocket or any streaming/real-time alerting, graph database,
microservices, Docker/Kubernetes deployment, LLMs, OAuth, a transaction simulator, or a calibration service.
Analysis is synchronous request/response. The project is **not deployed** anywhere.

## Machine learning

### XGBoost fraud model

Binary classifier trained on PaySim with a **strict temporal split by step** (train steps ≤ 520, validation 521–631,
test ≥ 632).

```text
n_estimators 150 · max_depth 5 · learning_rate 0.15 · subsample 0.8 · colsample_bytree 0.8
objective binary:logistic · eval_metric aucpr · random_state 42
```

The runtime feature vector has exactly **28 features** in the order of `Models/feature_list.json`:

| Group | Features |
|---|---|
| Transaction | `step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` |
| Balance | `orig_balance_change`, `dest_balance_change`, `amount_to_orig_balance`, `amount_to_dest_balance`, `orig_balance_error`, `dest_balance_error` |
| Amount / time | `log_amount`, `hour`, `day`, `is_night`, `hour_sin`, `hour_cos` |
| Sender history | `user_tx_count_before`, `user_amount_sum_before`, `user_avg_amount_before`, `amount_vs_user_avg`, `user_max_amount_before` |
| Type (one-hot) | `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER` |

### PaySim benchmark (synthetic)

Strict temporal test set, threshold 0.5. **Reproduced from the saved artifact** by `evaluation/run_evaluation.py`
(`evaluation/results/benchmark.json`).

| Metric | Value |
|---|---|
| Precision | 1.000000 |
| Recall | 0.999201 |
| F1 | 0.999600 |
| ROC-AUC | 1.000000 |
| PR-AUC | 0.999994 |
| Confusion matrix | TN 88,214 · FP 0 · FN 1 · TP 1,251 |

> These are **PaySim synthetic benchmark results**. They are not real-world or upay production performance. The test
> set is 89,466 rows with 1,252 fraud rows (**1.40% prevalence**; the whole dataset is 0.129%). It is the last 112
> steps of the simulation, where legitimate volume per step is ~15× lower than early on while fraud stays at ≈11
> per step. Precision and alert counts from this subset do not transfer to a lower-prevalence population.

### Why the near-perfect score should not be trusted as real-world accuracy

`orig_balance_error` (= old balance − amount − new balance) is by far the most important feature. On the test set it is
exactly 0 for ~99.4% of fraud rows but only ~21.5% of legitimate rows: PaySim's fraud agents move money with perfectly
consistent accounting while much legitimate traffic does not. The model learns this simulator-specific artifact.
Ablation studies (below, `evaluation/`) quantify how performance changes when it is removed. The sender-history features
are effectively inert on PaySim (99.85% of senders appear once; their SHAP importance is 0).

### Evaluation results (PaySim synthetic, same strict test set)

Measured by `evaluation/run_evaluation.py`; full tables in [`evaluation/results/evaluation_report.md`](evaluation/results/evaluation_report.md).
Ablations are retrained in memory and **remove derived columns only; the raw balance columns remain** in every variant except D and E.

| Model / variant | Precision | Recall | F1 | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|
| Production artifact | 1.0000 | 0.9992 | 0.9996 | 1.0000 | 0 | 1 |
| B: without `orig_balance_error` | 0.9505 | 0.9976 | 0.9735 | 0.9961 | 65 | 3 |
| C2: without all six balance-derived features | 0.6441 | 0.9976 | 0.7828 | 0.9907 | 690 | 3 |
| D: transaction type only | 0.0330 | 1.0000 | 0.0638 | 0.0519 | 36,727 | 0 |
| Logistic regression | 0.1968 | 0.9577 | 0.3264 | 0.8058 | 4,895 | 53 |
| Random forest | 1.0000 | 0.9992 | 0.9996 | 1.0000 | 0 | 1 |
| Rule: TRANSFER/CASH_OUT and account emptied | 0.0594 | 0.9585 | 0.1119 | 0.0575 | 19,004 | 52 |
| Amount threshold | 0.4817 | 0.1470 | 0.2252 | 0.0827 | 198 | 1,068 |

* A random forest reaches the **same confusion matrix** as the XGBoost model: PaySim fraud is very separable for tree models.
* Isolation Forest: ROC-AUC 0.8853, PR-AUC 0.3826 (anomaly ranking only).
* Risk levels on the test set (computed, not hard-coded): LOW 88,215 · MEDIUM 0 · HIGH 0 · CRITICAL 1,251; the HIGH+CRITICAL
  queue has precision 1.0000. That is **1,398 alerts per 100,000 transactions at the test set's 1.4% prevalence, ≈129 per
  100,000 re-weighted to the full dataset's 0.129%**. Zero observed false positives means a 95% upper bound of ≈3.4×10⁻⁵ on the
  false-positive rate, not zero.
* Hypothetical-cost threshold analysis: for the production model the threshold choice is immaterial (scores are almost binary);
  for less separable models low review thresholds are cheaper because a missed fraud costs its whole amount. Costs are
  **hypothetical assumptions** (no real cost data exists); the thresholds were **not** changed.
* Stress tests show the model is **not robust** to amount splitting (detection of at least one part falls to 52–97%, flagged parts to 5–50%)
  or to breaking exact balance accounting (detection falls to 22–27%), and it ignores the recipient's identity.
* Network connectivity patterns fire on *more* legitimate than fraudulent PaySim rows; they are context, not a detector.
* Demographic fairness cannot be evaluated (no demographic/KYC fields). Slices by type and amount band are in the report.
* Local latency (single machine, one process): feature engineering ≈7 ms, XGBoost + Isolation Forest ≈79 ms, SHAP ≈24 ms,
  behavioral queries ≈22 ms, network queries ≈7 ms, counterfactual (4 model runs) ≈348 ms; `analyze` over HTTP ≈180 ms mean
  (p95 ≈185 ms), ≈5.5 requests/s sequentially and ≈5.8 requests/s with 4 workers (CPU-bound). Not a production-scale claim.

### Isolation Forest

An anomaly **signal** only: `decision_function` (lower = more unusual; below 0 means the transaction looks unusual
compared with the rows it was trained on). It is shown to investigators but is **not** part of the production risk score,
because no production-safe calibration of it exists, and the notebook's test-set min-max normalisation is not reused.

### Risk engine

`risk_score = model score × 100`, mapped to application-defined bands:

| Risk score | Level | Recommended action |
|---:|---|---|
| 0–29 | LOW | ALLOW / MONITOR |
| 30–59 | MEDIUM | ADDITIONAL_VERIFICATION / MONITOR |
| 60–84 | HIGH | STEP_UP_VERIFICATION / MANUAL_REVIEW |
| 85–100 | CRITICAL | TEMPORARY_HOLD / ESCALATE_FOR_REVIEW |

The thresholds are application-defined, **not** calibrated probabilities. On the PaySim test set the model is almost
binary (scores near 0 or near 1), so MEDIUM and HIGH are essentially unused there; see the evaluation report. HIGH and
CRITICAL always require human review.

### SHAP, evidence, summary

* **SHAP** (`TreeExplainer`, raw-margin space) lists the top-5 feature attributions. It describes how the model used its
  inputs; it is not causal proof.
* **Evidence engine** collects model signal, anomaly value, SHAP attributions, behavioral and network observations.
* **Investigator summary** is a deterministic text built only from stored facts (no LLM). The model's output and the
  suggested **investigation recommendations** (analyst review, additional verification, recipient verification, account
  verification, temporary delay, transaction-limit review) are separate fields.

### Behavioral and network signals

Computed at analysis time from stored analyses and **persisted** (`behavioral_signals`, `network_signals` JSONB).
Point-in-time rules:

* only stored rows with a **strictly earlier `step`** are used, and only rows that already existed when the analysis ran
  (`created_at <= analysis time`); same-step and later-step rows never contribute;
* the stored result is returned by `GET .../network` and `GET .../behavior`, so later transactions cannot change a stored
  investigation. Records created before persistence existed are recomputed under the same rules and marked
  `source = "legacy_recomputed"`;
* behavioral velocity windows (5 min / 1 h / 24 h) are **wall-clock windows over storage time**, not PaySim steps.

Network signals (neutral wording, descriptive only): sender outgoing count / unique recipients / total outgoing amount;
recipient incoming count / unique senders / total incoming amount; previous transactions between the pair and
`NEW_RELATIONSHIP` / `EXISTING_RELATIONSHIP`; patterns `HIGH_RECIPIENT_CONNECTIVITY` and `HIGH_SENDER_CONNECTIVITY`
(thresholds `NETWORK_UNIQUE_SENDERS_THRESHOLD = 5`, `NETWORK_UNIQUE_RECIPIENTS_THRESHOLD = 5`, counts include the
current transaction). They never label an account as a mule or fraudulent.

**Model-feature history.** The caller supplies `historical_transactions`; rows with `step < current step` are used, rows
with the *same* step are accepted but **ignored**, rows with a *later* step are rejected (HTTP 422). *Difference from
training:* the original notebook computed the `user_*` features with `groupby(nameOrig).cumcount()` over step-sorted rows,
which also counted earlier rows of the same step. Model artifacts are unchanged; on PaySim this affects only the ~0.15% of
rows whose sender appears more than once (`evaluation/results/benchmark.json` reports the exact number and metrics).

### Counterfactual analysis

Re-runs the saved XGBoost model on four amount scenarios (100%, 75%, 50%, 25%), rebuilding all affected balances and all
28 features each time. Balance rules (derived from PaySim; the share of rows following each rule is in the audit notes):

* TRANSFER, CASH_OUT, DEBIT: sender balance falls by the amount, recipient balance rises by it;
* PAYMENT: sender balance falls; the merchant balance is left unchanged (PaySim does not track it);
* CASH_IN: sender balance **rises** by the amount and the counter-party balance falls.

Scenarios that would make a balance negative are marked invalid. Changes are measured against the **recomputed 100%
baseline** (recomputation makes the balance-error features 0, so it can differ from the stored score; the response
reports `baseline_matches_stored`). The best scenario must reduce the score by at least
`COUNTERFACTUAL_MIN_PROBABILITY_REDUCTION = 0.05`, otherwise it is `null`. Scenarios are not persisted.

The response separates **MODEL OUTPUT** from **INVESTIGATION RECOMMENDATION**. Counterfactuals are hypothetical model
responses, not causal explanations; a lower amount does not make a transaction legitimate.

## Analyst review workflow

`case_reviews` stores `status` (`OPEN`, `UNDER_REVIEW`, `DISMISSED`, `ESCALATED`, `CONFIRMED_SUSPICIOUS`),
`analyst_note`, `decision`, `reviewed_by`, `reviewed_at`, `created_at`, `updated_at`. A review **never changes** the stored
`fraud_probability`, `risk_level` or `recommended_action`. The dashboard review queue lists HIGH and CRITICAL transactions
whose review is missing or unresolved (`OPEN`, `UNDER_REVIEW`, `ESCALATED`), highest risk first.
`CONFIRMED_SUSPICIOUS` is an analyst judgement, not a fraud verdict.

## Security

* **API-key authentication** with two roles, `ANALYST` and `ADMIN` (ADMIN includes analyst rights). Keys come from
  `TRUSTX_ANALYST_API_KEY` / `TRUSTX_ADMIN_API_KEY`, are compared in constant time, and a role with no configured key
  cannot authenticate (fail closed). Optional `X-TrustX-User` labels an analyst in the audit log and review record; it is
  self-asserted text, **not** an authenticated identity.
* **Authorization:** `GET /health` is public; analysis, listing, summaries, signals, counterfactual and reviews need an
  analyst or admin key; `GET /api/v1/models/status` and `GET /api/v1/audit-logs` need an admin key.
* **Frontend:** the browser only calls a same-origin proxy route; the key is a server-only variable (`TRUSTX_API_KEY`),
  never in the client bundle, and the proxy exposes only analyst paths. The proxy is *not a login*: anyone who can open
  the frontend can use the analyst API through it. Acceptable for a local demo, not for production.
* **Input validation:** identifier length (≤100) and character set, `step` 0–100000, money 0–1e12, ≤500 history rows,
  unknown fields rejected, 256 KiB body cap; errors are clean JSON (400 malformed JSON, 401, 403, 404, 409, 413, 422, 429,
  500) without stack traces, paths or credentials.
* **Duplicate protection:** an identical transaction (SHA-256 of its own fields; history payload excluded) resubmitted
  within `TRUSTX_DUPLICATE_WINDOW_SECONDS` (default 24 h) returns the stored analysis with `is_duplicate = true`, so
  repeated submissions cannot pollute history. Very small race window under concurrent identical requests.
* **Audit log** (`audit_logs`): `TRANSACTION_ANALYZE`, `TRANSACTION_ANALYZE_DUPLICATE`, `TRANSACTION_VIEW`,
  `REVIEW_CREATE`, `REVIEW_UPDATE` with actor, role, resource, time and small metadata (never keys or note text).
* A small in-process rate limit protects `analyze` (`TRUSTX_RATE_LIMIT_PER_MINUTE`, default 120).

Not implemented: user accounts, password login, OAuth, per-user identity, TLS termination, key rotation, distributed rate
limiting. **If a database password was ever committed (an earlier `.env.example` contained one), rotate it.**

## API

All routes except `/health` require `X-API-Key`.

| Method | Path | Role |
|---|---|---|
| GET | `/health` | public |
| GET | `/api/v1/models/status` | ADMIN |
| GET | `/api/v1/audit-logs` | ADMIN |
| POST | `/api/v1/transactions/analyze` | ANALYST |
| GET | `/api/v1/transactions` (filters, `sort_by=risk_priority`, `review_queue=true`) | ANALYST |
| GET | `/api/v1/transactions/summary` · `/api/v1/risk/summary` | ANALYST |
| GET | `/api/v1/transactions/{id}` | ANALYST |
| GET | `/api/v1/transactions/{id}/behavior` · `/network` · `/counterfactual` | ANALYST |
| GET | `/api/v1/reviews` · `/api/v1/reviews/{id}` | ANALYST |
| POST / PATCH | `/api/v1/reviews/{id}` | ANALYST |

Interactive docs: `http://127.0.0.1:8000/docs`.

## Database

PostgreSQL via SQLAlchemy and Alembic (`backend/alembic/versions`):

| Migration | Content |
|---|---|
| 0001 / 0002 | `transaction_analyses` and indexes on `transaction_id`, `created_at`, `risk_level`, `type`, `recommended_action`, `fraud_probability`, `risk_score` |
| 0003 | `behavioral_signals` JSONB |
| 0004 | `network_signals` JSONB, `request_fingerprint` (+ index), composite indexes `(nameOrig, step)` and `(nameDest, step)` |
| 0005 | `case_reviews`, `audit_logs` |

## Setup

Requirements: Python 3.14 (what the tests and evaluation used), PostgreSQL, Node.js.

```powershell
# backend
cd D:\TrustX\backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd D:\TrustX
copy .env.example .env      # then fill in the placeholders (database password, two API keys)
.\backend\.venv\Scripts\alembic.exe upgrade head

# frontend
cd D:\TrustX\frontend
npm install
# frontend\.env.local (git-ignored):
#   NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
#   TRUSTX_API_KEY=<same value as TRUSTX_ANALYST_API_KEY>
npm run dev        # starts FastAPI on :8000 (if not running) and Next.js on :3000
```

`NEXT_PUBLIC_API_BASE_URL` is read by the frontend **server** (the proxy route); browsers call `/api/trustx/*`. Generate
keys with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Never commit `.env`, `.env.local`,
`node_modules`, `.venv` or `.next` (all git-ignored).

## Testing and evaluation

```powershell
cd D:\TrustX
.\backend\.venv\Scripts\python.exe -m unittest discover -s .\backend\tests -t .\backend -v
```

Last run: `Ran 167 tests ... OK` (167 passed, 0 failed, 0 skipped). Coverage by module: feature engineering and point-in-time history,
temporal correctness (future / same-step / stored-after-analysis exclusion, stored network result stability, legacy fallback),
behavioral engine, network analysis and persistence, counterfactual (all five transaction types, 100/75/50/25%, invalid balances,
real XGBoost, baseline consistency, interpretation), duplicate protection, API-key authentication and authorization, input
validation and error hygiene, analyst reviews (all statuses, note, reviewer, timestamps, model output immutability, review
queue), audit log, investigator summary and recommendations, risk levels, SHAP, evidence, persistence.
There are no automated frontend tests; the frontend is checked with `tsc --noEmit`, `npm run lint`, `npm run build` and
`scripts/e2e_demo.py`.

The DB-backed tests need the migrated PostgreSQL database from `.env`; they create their own rows and delete them.

Evaluation (separate from production, read-only on `Models/`; needs the PaySim CSV via `--csv` or `PAYSIM_CSV`):

```powershell
.\backend\.venv\Scripts\python.exe evaluation\run_evaluation.py --sections all
.\backend\.venv\Scripts\python.exe evaluation\run_evaluation.py --quick      # smoke test
.\backend\.venv\Scripts\python.exe scripts\profile_latency.py --n 200       # local latency profile
.\backend\.venv\Scripts\python.exe scripts\e2e_demo.py --frontend http://127.0.0.1:3000   # end-to-end check / demo data
```

Outputs are in `evaluation/results/` (`evaluation_report.md` plus JSON). See `evaluation/README.md`.

## Demo flow

1. Start the stack (`npm run dev`) and open `http://localhost:3000` — dashboard with risk counts and the review queue.
2. **Analyze** → load the *PaySim PAYMENT row*, analyze it (observed: LOW).
3. Load the *PaySim TRANSFER row (account emptied)*, analyze it (observed with the shipped model: CRITICAL).
4. Walk through risk assessment, SHAP, evidence, behavioral signals, network analysis, counterfactual (model output vs.
   investigation recommendation), investigator summary.
5. **Open full investigation**, create and update the analyst review; return to the dashboard to see the review status.
6. `scripts/e2e_demo.py` seeds demo data (including a fan-in example that shows `HIGH_RECIPIENT_CONNECTIVITY`).

MEDIUM and HIGH are not demonstrated: the shipped model's scores are almost binary on PaySim, so those levels are
rarely produced. Only claim levels you have actually observed.

## Dependency note (scikit-learn)

`Models/isolation_forest.joblib` was pickled with scikit-learn 1.6.1; the environment uses 1.9.1, which loads and
predicts but emits `InconsistentVersionWarning`. The artifact is intentionally **not** retrained or re-saved.
`requirements.txt` pins the versions the tests and evaluation ran on. For strict reproducibility install
scikit-learn 1.6.1 in an environment that supports it (not verified here).

## Limitations

* **Synthetic data.** Trained and evaluated only on PaySim; no real upay or MFS data; no real fraud labels.
* **Simulator artifact.** Performance depends strongly on exact balance accounting (`orig_balance_error`); see evaluation.
  Do not read the near-perfect benchmark as real-world accuracy.
* **Prevalence.** The benchmark test set is 1.40% fraud; real prevalence is unknown and likely different.
* **Uncalibrated score.** The model "fraud probability" is a model score, not a calibrated probability; risk thresholds
  are application-defined and the score is nearly binary on PaySim.
* **Inert history features.** Sender-history features carry no signal on PaySim; behavioral velocity signals can rarely be
  evaluated there. Behavioral windows are wall-clock based.
* **Weak network signals.** Two connectivity counts with a threshold of 5; popular merchants also have high fan-in.
* **Counterfactuals** are amount-only hypothetical model responses.
* **No** demographic/KYC fairness evaluation is possible with PaySim; no SIM-swap, device, agent, location,
  social-engineering or reversal data exists.
* **Security** is hackathon-level (see above). **Performance:** local single-machine numbers only, no production-scale claim.
* **Not** production-ready, not real-time streaming, not deployed, not validated on real data.
* The Isolation Forest artifact has the scikit-learn version mismatch described above.

## Project structure

```text
TrustX/
├── backend/app/{api,db,schemas,services}   FastAPI app (api/transactions.py, reviews.py, admin.py, risk.py)
├── backend/alembic/versions/               migrations 0001-0005
├── backend/tests/                          unittest suite (DB-backed tests included)
├── frontend/                               Next.js app, components, same-origin API proxy
├── Models/                                 trained artifacts (never modified by this code)
├── model_train/cpc_hackathon.ipynb         original training notebook (unchanged)
├── evaluation/                             evaluation suite and generated results
├── scripts/                                e2e_demo.py, profile_latency.py
└── docs/                                   ARCHITECTURE.md, MFS_SCENARIOS.md
```

## Model artifacts

`Models/final_xgb_model.json`, `isolation_forest.joblib`, `feature_list.json`, `model_config.json`,
`shap_feature_importance.csv` are the source of truth for inference and are never retrained or overwritten by this code.
`model_config.json` contains no `model_version`, so `model_version` is `null` in responses.
