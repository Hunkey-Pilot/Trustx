# Bangladesh MFS scenario coverage

TrustX was built for a mobile financial services (MFS) trust-and-risk hackathon track. This page states
honestly what the **current implementation** can and cannot say about common MFS abuse scenarios.

**Data reality.** The only data used is **PaySim**, a *synthetic* simulator of mobile-money transactions.
Each record has only: `step` (an hour index), `type` (CASH_IN / CASH_OUT / DEBIT / PAYMENT / TRANSFER),
`amount`, sender and recipient ids, and sender/recipient balances before and after the transaction.
There is **no** device, SIM, agent, location, KYC, session, customer-identity, reversal, or dispute data.
The model was trained on PaySim only and has never seen real upay or any real MFS data.

## What PaySim can and cannot represent

| PaySim can represent | PaySim cannot represent |
|---|---|
| Account-draining TRANSFER / CASH_OUT patterns (the simulator's fraud agents) | Devices, SIM cards, SIM swaps, device changes |
| Balance accounting before/after a transaction | Agents, agent ids, agent locations, commissions |
| Coarse activity over time (hourly steps) | Geography, customer KYC level, account age, demographics |
| Recipient fan-in / sender fan-out counts | Reversals, refunds, disputes, complaints |
| Amount and transaction-type patterns | Social-engineering context (calls, OTP sharing, phishing) |
| | Per-customer history: 99.85% of PaySim senders appear exactly once |

**What real upay data would add:** device and SIM-change events, agent id and location, KYC tier and account age,
reversal/refund records, customer complaints, confirmed-fraud labels, and genuine per-customer histories.
None of this exists in the repository and none of it is simulated here.

## Scenario by scenario

| Scenario | What TrustX can detect with current transaction data | Evidence it can provide | What PaySim supports | What needs real MFS data |
|---|---|---|---|---|
| **Unusual transfer** | Transfers whose balance pattern and amount look like the PaySim fraud pattern; a transfer far above the sender's prior average (only if the sender has history) | Model score, SHAP attributions, amount vs prior average, relationship status (new/existing) | Yes, this is what the model was trained on. Note the strong dependence on exact balance accounting (see README limitations) | Real customer histories and limits; real labels |
| **Agent cash-out abuse** | CASH_OUT transactions that look like account-draining; nothing agent-specific | Model score and attributions for the CASH_OUT row | PaySim has CASH_OUT rows but **no agent id, location or agent history** | Agent ids, agent-level velocity, agent location vs customer location, commission patterns |
| **Mule-like transaction behavior** | Descriptive connectivity only: a recipient receiving from many distinct senders, or a sender paying many distinct recipients (threshold 5, counts include the current transaction) | `HIGH_RECIPIENT_CONNECTIVITY` / `HIGH_SENDER_CONNECTIVITY` patterns, unique sender/recipient counts, relationship status | Recipient ids exist, but on PaySim merchants and popular accounts also have high fan-in, so this signal is weak. See `evaluation/results/evaluation_report.md` | Account metadata (age, KYC), fast in/out money movement over time, linked devices, confirmed mule labels. TrustX never labels an account as a mule |
| **Fake reversal / refund pattern** | **Nothing.** There is no reversal or refund transaction type | None | Not represented | Reversal and refund records linked to original transactions, dispute data |
| **Social engineering** | **Nothing directly.** A victim-authorised transfer can look normal in transaction data. At most the model may flag an unusual amount or new recipient | Possibly: new relationship, amount vs history (if history exists) | Not represented (no session, call or OTP context) | Session and device context, call/SMS signals, behavioural biometrics, victim reports |
| **SIM-swap-related activity** | **Nothing.** No SIM or device information is available | None | Not represented | SIM-change / re-registration events, device fingerprint changes, time since SIM change |

## What this means for claims

* TrustX **can** show how a transaction scores on a PaySim-trained model, why (SHAP), and which simple relationship
  or behaviour signals accompany it.
* TrustX **cannot** claim to detect SIM swap, social engineering, agent abuse, fake reversals or identity fraud.
  Those are described here only as *requirements for real data*.
* Network patterns are descriptive and never label an account as a mule or a transaction as confirmed fraud.
* HIGH and CRITICAL assessments are routed to a human analyst; the analyst decision is stored separately from the
  model output.
