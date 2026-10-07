# TrustX architecture (as implemented)

This diagram shows only components that exist in the repository. There is **no** Redis, Kafka, WebSocket,
streaming pipeline, simulator service, model-calibration service, or background worker. Analysis is a
synchronous request/response call.

```mermaid
flowchart TD
    Browser[Browser: Next.js pages] -->|same-origin /api/trustx/*| Proxy[Next.js route handler proxy<br/>adds server-side API key<br/>allow-lists analyst paths]
    Proxy -->|HTTP + X-API-Key| API[FastAPI backend]
    CLI[scripts/e2e_demo.py and tests] -->|HTTP + X-API-Key| API

    API --> Auth[API-key auth<br/>ANALYST / ADMIN roles]
    Auth --> Analyze[POST /transactions/analyze]
    Analyze --> Dup{Duplicate fingerprint<br/>in window?}
    Dup -->|yes| Stored[(Return stored analysis)]
    Dup -->|no| FE[Feature engineering<br/>exact 28 features<br/>history: earlier steps only]
    FE --> XGB[XGBoost model score]
    FE --> IF[Isolation Forest anomaly value<br/>not part of risk score]
    XGB --> Risk[Risk engine<br/>score x 100 to LOW/MEDIUM/HIGH/CRITICAL]
    FE --> SHAP[SHAP TreeExplainer]
    Analyze --> Sig[Behavioral + network signals<br/>earlier steps already stored]
    Risk --> Ev[Evidence engine]
    SHAP --> Ev
    Sig --> Ev
    Ev --> Sum[Deterministic investigator summary<br/>+ investigation recommendations]
    Sum --> PG[(PostgreSQL)]

    PG --- T1[transaction_analyses<br/>incl. behavioral_signals and network_signals JSONB]
    PG --- T2[case_reviews]
    PG --- T3[audit_logs]

    API --> CF[GET /counterfactual<br/>re-runs the real model on amount scenarios]
    API --> Rev[/reviews: analyst workflow/]
    Rev --> T2
    Auth --> Audit[Audit logging] --> T3
```

## Request flow for `POST /api/v1/transactions/analyze`

1. API key is checked (ANALYST or ADMIN); an in-process rate limit applies.
2. Input is validated (length limits, ranges, no unknown fields); body size is capped at 256 KiB.
3. A SHA-256 fingerprint of the transaction's own fields is looked up; an identical transaction inside the duplicate
   window (default 24 h) returns the stored analysis with `is_duplicate = true`.
4. The 28 features are built (history rows with `step < current step` only), XGBoost and the Isolation Forest are scored,
   SHAP is computed, the risk engine maps the model score to a level and recommended action.
5. Behavioral and network signals are computed from stored analyses with a strictly earlier `step` that already existed
   when the analysis started. **They are persisted with the analysis**, so later transactions never change them.
6. The evidence and a deterministic investigator summary are built and the whole analysis is stored.
7. An audit entry is written (`TRANSACTION_ANALYZE`).

## Human review

HIGH and CRITICAL assessments appear in the dashboard review queue. An analyst creates and updates a review
(`OPEN`, `UNDER_REVIEW`, `DISMISSED`, `ESCALATED`, `CONFIRMED_SUSPICIOUS`) through `/api/v1/reviews`. The review is
stored separately from the model output; it never changes `fraud_probability`, `risk_level` or `recommended_action`.

## Not in the architecture (deliberately)

Redis, Kafka/Celery, WebSocket or other streaming, graph databases, feature stores, microservices, LLMs, OAuth,
model calibration/versioning services, a transaction simulator, Docker/Kubernetes deployment.
