# TrustX

### AI Trust & Risk Intelligence Platform

TrustX is an AI-powered transaction trust and risk analysis platform designed to identify potentially suspicious financial transactions using machine learning, behavioral signals, network analysis, explainable AI, and counterfactual analysis.

The platform combines an XGBoost fraud classifier, Isolation Forest anomaly detection, SHAP-based explanations, risk scoring, evidence generation, PostgreSQL persistence, network analysis, and a Next.js investigation dashboard into a single system.

> **Hackathon:** AI Hackathon 2026 — DIU CPC × upay
> **Track:** Trust & Risk

---

## 🚀 Key Features

* AI-based transaction fraud probability prediction
* Isolation Forest anomaly detection
* Application-level risk scoring
* Risk levels:

  * LOW
  * MEDIUM
  * HIGH
  * CRITICAL
* Recommended actions for risk levels
* SHAP-based individual transaction explanation
* Evidence generation
* Behavioral and velocity signals
* Sender/recipient network analysis
* Mule-network pattern detection
* Counterfactual analysis
* Transaction history and persistence
* Filtering, sorting, and pagination
* Risk and transaction summaries
* Interactive investigation dashboard
* Next.js frontend
* FastAPI backend
* PostgreSQL database

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │     Next.js UI      │
                         │                     │
                         │ Dashboard           │
                         │ Transactions        │
                         │ Investigation       │
                         └──────────┬──────────┘
                                    │
                                    │ REST API
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │      Backend        │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
       ┌─────────────┐       ┌─────────────┐      ┌──────────────┐
       │   XGBoost   │       │  Isolation  │      │  PostgreSQL  │
       │ Fraud Model │       │   Forest    │      │   Database   │
       └──────┬──────┘       └──────┬──────┘      └──────────────┘
              │                     │
              └──────────┬──────────┘
                         ▼
                ┌──────────────────┐
                │   Risk Engine    │
                └────────┬─────────┘
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
       ┌──────────────┐       ┌──────────────┐
       │ SHAP Engine  │       │ Evidence     │
       │              │       │ Engine       │
       └──────────────┘       └──────────────┘
              │
              ▼
       ┌─────────────────────────────┐
       │ Behavioral / Network /      │
       │ Counterfactual Analysis     │
       └─────────────────────────────┘
```

---

# 📁 Project Structure

```text
TrustX/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── feature_engineering.py
│   │   │   ├── risk_engine.py
│   │   │   ├── shap_engine.py
│   │   │   ├── evidence_engine.py
│   │   │   ├── behavioral_signals.py
│   │   │   ├── network_engine.py
│   │   │   └── counterfactual_engine.py
│   │   └── ...
│   │
│   ├── tests/
│   ├── alembic/
│   ├── requirements.txt
│   └── ...
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── transactions/
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   ├── lib/
│   │   └── api.ts
│   ├── package.json
│   └── ...
│
├── Models/
│   ├── final_xgb_model.json
│   ├── isolation_forest.joblib
│   ├── feature_list.json
│   ├── model_config.json
│   └── shap_feature_importance.csv
│
├── model_train/
│   └── cpc_hackathon.ipynb
│
├── .gitignore
└── README.md
```

---

# 🧠 Machine Learning

## XGBoost Fraud Classifier

TrustX uses a trained XGBoost binary classification model to estimate the probability that a transaction belongs to the fraud class.

### Model Configuration

```text
Model: XGBoost
Task: Binary Fraud Classification
Dataset: PaySim
Validation: Strict Temporal Split

n_estimators: 150
max_depth: 5
learning_rate: 0.15
subsample: 0.8
colsample_bytree: 0.8

objective: binary:logistic
eval_metric: aucpr
random_state: 42
```

## Model Features

The model uses exactly **28 features**.

### Transaction Features

```text
step
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

### Balance Features

```text
orig_balance_change
dest_balance_change
amount_to_orig_balance
amount_to_dest_balance
orig_balance_error
dest_balance_error
```

### Time Features

```text
log_amount
hour
day
is_night
hour_sin
hour_cos
```

### Sender Behavioral Features

```text
user_tx_count_before
user_amount_sum_before
user_avg_amount_before
amount_vs_user_avg
user_max_amount_before
```

### Transaction Type Features

```text
type_CASH_IN
type_CASH_OUT
type_DEBIT
type_PAYMENT
type_TRANSFER
```

The runtime feature vector follows the exact order defined in:

```text
Models/feature_list.json
```

---

# 📊 Benchmark Performance

The XGBoost model was evaluated using a strict temporal split on the **PaySim synthetic dataset**.

| Metric    |    Score |
| --------- | -------: |
| Precision | 1.000000 |
| Recall    | 0.999201 |
| F1 Score  | 0.999600 |
| ROC-AUC   | 1.000000 |
| PR-AUC    | 0.999994 |

### Confusion Matrix

```text
TN = 88,214
FP = 0
FN = 1
TP = 1,251
```

> These results are benchmark results on the PaySim synthetic dataset. They should not be interpreted as real-world or production performance on upay or any live financial system.

---

# 🔍 Anomaly Detection

TrustX also uses an **Isolation Forest** model to generate an anomaly signal.

The Isolation Forest was trained using normal transactions from the chronological training portion of the PaySim dataset.

The anomaly output is used as an additional signal and is not treated as proof of fraud.

---

# ⚠️ Risk Engine

TrustX converts the XGBoost fraud probability into an application-defined risk score.

Risk levels:

| Risk Score | Level    | Recommended Action                   |
| ---------: | -------- | ------------------------------------ |
|       0–29 | LOW      | ALLOW / MONITOR                      |
|      30–59 | MEDIUM   | ADDITIONAL_VERIFICATION / MONITOR    |
|      60–84 | HIGH     | STEP_UP_VERIFICATION / MANUAL_REVIEW |
|     85–100 | CRITICAL | TEMPORARY_HOLD / ESCALATE_FOR_REVIEW |

The risk score is an **application-defined decision-support score**. It is not a calibrated probability and should not be interpreted as a universal fraud threshold.

---

# 🧩 Explainable AI

## SHAP

TrustX uses SHAP to explain individual XGBoost predictions.

For each analyzed transaction, the system can provide:

* Feature
* Feature value
* SHAP value
* Direction of contribution

Example:

```text
Feature: amount
Value: 100
SHAP Value: -1.14
Direction: decreases fraud prediction
```

Positive SHAP values push the model toward the positive/fraud class, while negative values push it away.

SHAP output is a model attribution and should not be interpreted as causal proof.

---

# 🧾 Evidence Engine

The Evidence Engine combines multiple model and analytical signals into structured evidence.

Supported evidence categories include:

```text
MODEL_SIGNAL
ANOMALY_SIGNAL
MODEL_ATTRIBUTION
NETWORK_SIGNAL
```

The system also returns a separate risk assessment.

Evidence is designed to support investigation and decision-making.

TrustX does **not** describe individual signals as confirmed fraud or proof of fraudulent activity.

---

# 👤 Behavioral Analysis

TrustX analyzes historical transaction behavior to identify unusual activity.

Signals include:

* Transaction velocity
* Previous sender activity
* Historical transaction amounts
* Recipient behavior
* Amount deviation from previous behavior
* Recipient-related activity

These signals provide additional context around the model prediction.

---

# 🕸️ Network / Mule-Network Analysis

TrustX analyzes sender-recipient relationships using transaction history.

### Sender Signals

```text
outgoing_transaction_count
unique_recipient_count
total_outgoing_amount
```

### Recipient Signals

```text
incoming_transaction_count
unique_sender_count
total_incoming_amount
```

### Relationship Signals

```text
previous_transaction_count
previous_transaction_amount
relationship_status
```

Relationship status:

```text
NEW_RELATIONSHIP
EXISTING_RELATIONSHIP
```

### Network Patterns

TrustX currently detects simple network patterns such as:

```text
HIGH_RECIPIENT_CONNECTIVITY
HIGH_SENDER_CONNECTIVITY
```

These patterns are network-based risk signals and do not automatically classify an account as a mule or fraudulent.

---

# 🔄 Counterfactual Analysis

TrustX provides hypothetical transaction scenarios to understand how the model prediction may change if the transaction amount were different.

The system evaluates:

```text
100% of original amount
75% of original amount
50% of original amount
25% of original amount
```

For each scenario, the system recalculates the relevant balance values and runs the existing feature engineering and XGBoost model.

The best hypothetical scenario is selected only when the fraud probability reduction meets the configured threshold.

```text
COUNTERFACTUAL_MIN_PROBABILITY_REDUCTION = 0.05
```

Counterfactual results are hypothetical model scenarios and do not guarantee that changing a transaction amount would prevent fraud.

---

# 🗄️ Database

TrustX uses PostgreSQL for transaction analysis persistence.

Stored information includes analysis results such as:

* Transaction information
* Fraud probability
* Anomaly signal
* Risk score
* Risk level
* Recommended action
* SHAP explanation
* Evidence
* Behavioral signals
* Network-related information where applicable
* Timestamps

Database migrations are managed using Alembic.

---

# 🔌 API Endpoints

## Health

```http
GET /health
```

## Model Status

```http
GET /api/v1/models/status
```

## Analyze Transaction

```http
POST /api/v1/transactions/analyze
```

## Transaction History

```http
GET /api/v1/transactions
```

## Transaction Summary

```http
GET /api/v1/transactions/summary
```

## Risk Summary

```http
GET /api/v1/risk/summary
```

## Transaction Details

```http
GET /api/v1/transactions/{transaction_id}
```

## Network Analysis

```http
GET /api/v1/transactions/{transaction_id}/network
```

## Counterfactual Analysis

```http
GET /api/v1/transactions/{transaction_id}/counterfactual
```

Additional endpoints are available for explanation, evidence, behavioral analysis, and risk retrieval.

---

# 🖥️ Frontend

The TrustX frontend is built using:

```text
Next.js
TypeScript
React
CSS
```

The frontend provides:

### Dashboard

* Transaction summary
* Risk distribution
* Average fraud probability
* Average risk score
* Recent transactions

### Transaction Investigation

For an individual transaction, investigators can view:

* Transaction details
* Fraud probability
* Risk score
* Risk level
* Recommended action
* Anomaly signal
* SHAP explanation
* Evidence
* Behavioral signals
* Network analysis
* Counterfactual analysis

---

# ⚙️ Installation

## Requirements

Make sure the following are installed:

```text
Python 3.x
Node.js
npm
PostgreSQL
```

---

# 🔧 Backend Setup

Go to the backend directory:

```powershell
cd D:\TrustX\backend
```

Create and activate a virtual environment if needed:

```powershell
python -m venv .venv
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create:

```text
backend/.env
```

Example:

```env
DATABASE_URL=postgresql+psycopg://postgres:<YOUR_PASSWORD>@localhost:5432/trustx
```

Do not commit `.env` to GitHub.

A safe template can be provided in:

```text
.env.example
```

---

# 🗃️ Database Migration

Run:

```powershell
cd D:\TrustX\backend
python.exe -m alembic -c alembic.ini upgrade head
```

---

# ▶️ Start Backend

```powershell
cd D:\TrustX\backend
python.exe -m uvicorn app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🎨 Frontend Setup

Go to:

```powershell
cd D:\TrustX\frontend
```

Install dependencies:

```powershell
npm install
```

Create:

```text
frontend/.env.local
```

Add:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Start the frontend:

```powershell
npm run dev
```

Frontend:

```text
http://localhost:3000
```

---

# 🧪 Testing

Run the backend test suite:

```powershell
cd D:\TrustX\backend
python.exe -m unittest discover -s .\tests -t .\ -v
```

The project includes automated tests covering areas such as:

* Feature engineering
* Model inference
* Risk engine
* SHAP explanation
* Evidence engine
* Database persistence
* Transaction history
* Behavioral signals
* Network analysis
* Counterfactual analysis
* API behavior

---

# 🔄 Typical Workflow

```text
1. User submits transaction
          ↓
2. Feature Engineering
          ↓
3. XGBoost Fraud Prediction
          ↓
4. Isolation Forest Anomaly Signal
          ↓
5. Risk Engine
          ↓
6. SHAP Explanation
          ↓
7. Evidence Engine
          ↓
8. Behavioral Analysis
          ↓
9. Network Analysis
          ↓
10. Counterfactual Analysis
          ↓
11. PostgreSQL Persistence
          ↓
12. Next.js Investigation Dashboard
```

---

# 🔒 Security & Limitations

TrustX is currently a hackathon/prototype system and is not intended to be deployed directly as a production financial fraud prevention system.

Current limitations include:

* No production authentication/authorization layer
* No production-grade deployment configuration
* No Docker deployment
* No real-time WebSocket alerting
* Models were trained on PaySim synthetic data
* Model performance has not been validated on real upay production data
* Network analysis uses simplified relationship signals
* Counterfactual analysis is limited to transaction amount scenarios
* Risk thresholds are application-defined
* Anomaly signal is not converted into a calibrated fraud probability

---

# ⚠️ Important ML Disclaimer

The model outputs are decision-support signals.

A high fraud probability, anomaly signal, SHAP attribution, behavioral signal, or network pattern does not independently establish that a transaction is fraudulent.

TrustX is designed to help investigators identify and understand potentially risky transactions.

---

# 📌 Model Artifacts

The trained artifacts are stored under:

```text
Models/
```

Important files:

```text
final_xgb_model.json
isolation_forest.joblib
feature_list.json
model_config.json
shap_feature_importance.csv
```

These artifacts are treated as the source of truth for runtime inference.

The application does not retrain the models during normal API execution.

---

# 🏆 Hackathon Context

TrustX was developed for:

**AI Hackathon 2026**

Organized through:

**DIU CPC × upay**

Track:

**Trust & Risk**

The system demonstrates how machine learning, explainable AI, behavioral analysis, network analysis, and transaction intelligence can be combined into a unified financial trust and risk analysis platform.

---

# 👥 Project

**TrustX — AI Trust & Risk Intelligence Platform**

Built with:

```text
Python
FastAPI
XGBoost
Scikit-learn
SHAP
PostgreSQL
SQLAlchemy
Alembic
Next.js
TypeScript
React
```

---

## ⭐ Project Vision

TrustX aims to move beyond simple binary fraud classification by combining:

**Prediction + Anomaly Detection + Explainability + Behavioral Intelligence + Network Intelligence + Counterfactual Reasoning**

into one investigator-friendly platform.

The goal is not only to answer:

> "Is this transaction suspicious?"

but also:

> "Why does the model consider it risky?"

> "What behavioral or network signals support the assessment?"

> "How is this transaction connected to other activity?"

> "What hypothetical change could reduce the model's predicted risk?"
