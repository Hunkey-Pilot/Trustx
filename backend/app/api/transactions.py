from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from fastapi.params import Param
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.repository import (
    create_transaction_analysis,
    find_recent_duplicate,
    get_review,
    get_transaction_analysis,
    list_transaction_analyses,
    request_fingerprint,
    transaction_analysis_to_response,
    transaction_summary,
)
from app.db.session import get_db
from app.schemas.transactions import (
    MAX_TRANSACTION_ID_LENGTH,
    TRANSACTION_ID_PATTERN,
    BehavioralSignals,
    CounterfactualResponse,
    NetworkSignals,
    PersistedTransactionResponse,
    ShapExplanation as ShapExplanationSchema,
    TransactionAnalyzeRequest,
    TransactionAnalyzeResponse,
    TransactionListResponse,
    TransactionSummaryResponse,
    TransactionType,
)
from app.security import Actor, actor_or_none, require_analyst
from app.services.audit_service import record_audit
from app.services.behavioral_engine import BehavioralEngine
from app.services.counterfactual_engine import CounterfactualEngine
from app.services.feature_engineering import (
    FeatureEngineeringError,
    create_model_features,
)
from app.services.investigator_summary import build_investigator_summary
from app.services.model_service import ModelInferenceError
from app.services.network_engine import NetworkEngine
from app.services.rate_limit import enforce_analyze_rate_limit
from app.services.risk_engine import RiskEngineError
from app.services.shap_service import ShapExplanation as ShapServiceExplanation


logger = logging.getLogger("trustx.api")

router = APIRouter(
    prefix="/api/v1/transactions",
    tags=["transactions"],
    dependencies=[Depends(require_analyst)],
)

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RecommendedAction = Literal[
    "ALLOW / MONITOR",
    "ADDITIONAL_VERIFICATION / MONITOR",
    "STEP_UP_VERIFICATION / MANUAL_REVIEW",
    "TEMPORARY_HOLD / ESCALATE_FOR_REVIEW",
]
SortField = Literal["created_at", "fraud_probability", "risk_score", "amount", "risk_priority"]
SortOrder = Literal["asc", "desc"]
ReviewFilter = Literal[
    "UNREVIEWED", "OPEN", "UNDER_REVIEW", "DISMISSED", "ESCALATED", "CONFIRMED_SUSPICIOUS"
]
TransactionIdPath = Annotated[
    str,
    Path(min_length=1, max_length=MAX_TRANSACTION_ID_LENGTH, pattern=TRANSACTION_ID_PATTERN),
]


def _duplicate_window_seconds() -> int:
    try:
        return int(os.getenv("TRUSTX_DUPLICATE_WINDOW_SECONDS", "86400"))
    except ValueError:
        return 86400


def _record_to_request(record: Any) -> TransactionAnalyzeRequest:
    try:
        return TransactionAnalyzeRequest.model_validate(
            {
                "step": record.step,
                "type": record.transaction_type,
                "amount": record.amount,
                "nameOrig": record.name_orig,
                "oldbalanceOrg": record.old_balance_org,
                "newbalanceOrig": record.new_balance_orig,
                "nameDest": record.name_dest,
                "oldbalanceDest": record.old_balance_dest,
                "newbalanceDest": record.new_balance_dest,
                "historical_transactions": record.historical_transactions or [],
            }
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The stored transaction no longer satisfies input limits and cannot be re-evaluated",
        ) from exc


def _load_record_or_404(db: Session, transaction_id: str):
    try:
        record = get_transaction_analysis(db, transaction_id)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction record is temporarily unavailable",
        ) from exc
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    return record


def _summary_input(record_response: PersistedTransactionResponse) -> dict[str, Any]:
    return record_response.model_dump(mode="json")


def _analysis_from_persisted(
    persisted: PersistedTransactionResponse,
    is_duplicate: bool,
) -> TransactionAnalyzeResponse:
    response = TransactionAnalyzeResponse.model_validate(persisted.model_dump())
    response.is_duplicate = is_duplicate
    response.investigator_summary = build_investigator_summary(_summary_input(persisted))
    return response


@router.post(
    "/analyze",
    response_model=TransactionAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze one transaction with the trained models",
    description=(
        "Builds the exact 28-feature model vector from the transaction and "
        "historical sender context, then returns the trained XGBoost probability "
        "and raw Isolation Forest anomaly signal. Resubmitting an identical "
        "transaction within the duplicate window returns the stored analysis "
        "(`is_duplicate = true`) instead of creating a new record."
    ),
)
def analyze_transaction(
    request: Request,
    transaction: TransactionAnalyzeRequest,
    db: Session = Depends(get_db),
    actor: Actor = Depends(enforce_analyze_rate_limit),
) -> TransactionAnalyzeResponse:
    actor = actor_or_none(actor)
    analysis_time = datetime.now(timezone.utc)

    if isinstance(db, Session):
        try:
            duplicate = find_recent_duplicate(
                db, request_fingerprint(transaction), _duplicate_window_seconds()
            )
        except SQLAlchemyError:
            db.rollback()
            duplicate = None
        if duplicate is not None:
            record_audit(
                db,
                actor,
                "TRANSACTION_ANALYZE_DUPLICATE",
                "transaction",
                duplicate.transaction_id,
                {"risk_level": duplicate.risk_level},
            )
            return _analysis_from_persisted(
                transaction_analysis_to_response(duplicate), is_duplicate=True
            )

    historical_context = pd.DataFrame(
        [item.model_dump() for item in transaction.historical_transactions],
        columns=["step", "amount", "nameOrig"],
    )
    transaction_values = transaction.model_dump(exclude={"historical_transactions"})

    try:
        model_loader = request.app.state.model_loader
        features = create_model_features(
            transaction=transaction_values,
            historical_context=historical_context,
            feature_list=model_loader.loaded.feature_list,
        )
        prediction = request.app.state.model_service.predict(features)
    except (FeatureEngineeringError, ModelInferenceError, RiskEngineError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AttributeError as exc:
        logger.error("Model services are not initialised")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained model services are unavailable",
        ) from exc

    risk_result = request.app.state.risk_engine.calculate(
        fraud_probability=prediction.fraud_probability,
        anomaly_signal=prediction.anomaly_signal,
    )
    shap_result: ShapServiceExplanation = request.app.state.shap_service.explain(features)
    if shap_result.error:
        logger.warning("SHAP explanation unavailable: %s", shap_result.error)
    transaction_id = f"req_{uuid4().hex}"
    behavioral_engine = getattr(request.app.state, "behavioral_engine", None)
    network_engine = getattr(request.app.state, "network_engine", None)
    if isinstance(db, Session) and behavioral_engine is not None:
        behavioral_signals = behavioral_engine.calculate(
            db,
            transaction,
            transaction_id=transaction_id,
            analysis_time=analysis_time,
        )
    else:
        behavioral_signals = BehavioralSignals(
            status="unavailable",
            error="Behavioral signals unavailable: database session or engine not configured",
        )
    if isinstance(db, Session) and network_engine is not None:
        network_signals = network_engine.calculate(
            db,
            transaction,
            transaction_id=transaction_id,
            analysis_time=analysis_time,
        )
    else:
        network_signals = NetworkSignals(
            status="unavailable",
            error="Network signals unavailable: database session or engine not configured",
        )
    evidence_result = request.app.state.evidence_engine.aggregate(
        fraud_probability=prediction.fraud_probability,
        anomaly_signal=prediction.anomaly_signal,
        shap_explanation=shap_result,
        risk_result=risk_result,
        behavioral_signals=behavioral_signals,
        network_signals=network_signals,
    )
    model_version = model_loader.loaded.model_config.get("model_version")
    response = TransactionAnalyzeResponse(
        transaction_id=transaction_id,
        fraud_probability=prediction.fraud_probability,
        anomaly_signal=prediction.anomaly_signal,
        risk_score=risk_result.risk_score,
        risk_level=risk_result.risk_level,
        recommended_action=risk_result.recommended_action,
        explanation=ShapExplanationSchema(
            status=shap_result.status,
            method=shap_result.method,
            type=shap_result.explanation_type,
            output_space=shap_result.output_space,
            base_value=shap_result.base_value,
            top_features=[
                {
                    "feature": item.feature,
                    "value": item.value,
                    "shap_value": item.shap_value,
                    "direction": item.direction,
                }
                for item in shap_result.top_features
            ],
            error="SHAP explanation is currently unavailable" if shap_result.error else None,
        ),
        evidence={
            "status": evidence_result.status,
            "items": [
                {
                    "category": item.category,
                    "type": item.type,
                    "value": item.value,
                    "description": item.description,
                    "feature": item.feature,
                    "shap_value": item.shap_value,
                    "direction": item.direction,
                }
                for item in evidence_result.items
            ],
            "risk_assessment": {
                "risk_score": evidence_result.risk_assessment.risk_score,
                "risk_level": evidence_result.risk_assessment.risk_level,
                "recommended_action": evidence_result.risk_assessment.recommended_action,
            },
        },
        behavioral_signals=behavioral_signals,
        network_signals=network_signals,
        model_version=model_version,
        feature_count=prediction.feature_count,
    )
    response.investigator_summary = build_investigator_summary(
        {
            **transaction_values,
            **response.model_dump(mode="json", exclude={"investigator_summary"}),
        }
    )

    if isinstance(db, Session):
        try:
            create_transaction_analysis(db, transaction, response, created_at=analysis_time)
        except SQLAlchemyError as exc:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Analysis completed but could not be persisted",
            ) from exc
        record_audit(
            db,
            actor,
            "TRANSACTION_ANALYZE",
            "transaction",
            transaction_id,
            {"risk_level": response.risk_level, "type": transaction.type},
        )

    return response


@router.get(
    "",
    response_model=TransactionListResponse,
    summary="List persisted transaction analyses",
)
def list_transactions(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    risk_level: RiskLevel | None = Query(None),
    recommended_action: RecommendedAction | None = Query(None),
    type: TransactionType | None = Query(None),
    transaction_id: str | None = Query(None, min_length=1, max_length=MAX_TRANSACTION_ID_LENGTH),
    created_after: datetime | None = Query(None),
    created_before: datetime | None = Query(None),
    min_fraud_probability: float | None = Query(None, ge=0, le=1),
    max_fraud_probability: float | None = Query(None, ge=0, le=1),
    min_risk_score: float | None = Query(None, ge=0, le=100),
    max_risk_score: float | None = Query(None, ge=0, le=100),
    sort_by: SortField = Query("created_at"),
    sort_order: SortOrder = Query("desc"),
    review_queue: bool = Query(
        False,
        description=(
                    "Only HIGH/CRITICAL transactions with no review or an unresolved one "
                    "(OPEN, UNDER_REVIEW, ESCALATED)."
                ),
    ),
    review_status: ReviewFilter | None = Query(None),
) -> TransactionListResponse:
    risk_level = _resolve_query_default(risk_level, None)
    recommended_action = _resolve_query_default(recommended_action, None)
    type = _resolve_query_default(type, None)
    transaction_id = _resolve_query_default(transaction_id, None)
    created_after = _resolve_query_default(created_after, None)
    created_before = _resolve_query_default(created_before, None)
    min_fraud_probability = _resolve_query_default(min_fraud_probability, None)
    max_fraud_probability = _resolve_query_default(max_fraud_probability, None)
    min_risk_score = _resolve_query_default(min_risk_score, None)
    max_risk_score = _resolve_query_default(max_risk_score, None)
    sort_by = _resolve_query_default(sort_by, "created_at")
    sort_order = _resolve_query_default(sort_order, "desc")
    review_queue = _resolve_query_default(review_queue, False)
    review_status = _resolve_query_default(review_status, None)
    if (
        min_fraud_probability is not None
        and max_fraud_probability is not None
        and min_fraud_probability > max_fraud_probability
    ) or (
        min_risk_score is not None
        and max_risk_score is not None
        and min_risk_score > max_risk_score
    ) or (
        created_after is not None
        and created_before is not None
        and created_after > created_before
    ):
        raise HTTPException(status_code=422, detail="Filter lower bounds cannot exceed upper bounds")
    try:
        return list_transaction_analyses(
            db,
            page=page,
            page_size=page_size,
            risk_level=risk_level,
            recommended_action=recommended_action,
            transaction_type=type,
            transaction_id=transaction_id,
            created_after=created_after,
            created_before=created_before,
            min_fraud_probability=min_fraud_probability,
            max_fraud_probability=max_fraud_probability,
            min_risk_score=min_risk_score,
            max_risk_score=max_risk_score,
            sort_by=sort_by,
            sort_order=sort_order,
            review_queue=review_queue,
            review_status=review_status,
        )
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction records are temporarily unavailable",
        ) from exc


def _resolve_query_default(value, default):
    return default if isinstance(value, Param) else value


@router.get("/summary", response_model=TransactionSummaryResponse)
def transaction_summary_endpoint(
    db: Session = Depends(get_db),
) -> TransactionSummaryResponse:
    try:
        return transaction_summary(db)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction summary is temporarily unavailable",
        ) from exc


@router.get(
    "/{transaction_id}/behavior",
    response_model=BehavioralSignals,
    summary="Retrieve behavioral signals for a persisted transaction",
)
def transaction_behavior(
    transaction_id: TransactionIdPath,
    db: Session = Depends(get_db),
) -> BehavioralSignals:
    record = _load_record_or_404(db, transaction_id)
    stored = record.behavioral_signals
    if isinstance(stored, dict) and stored.get("status") == "available":
        signals = BehavioralSignals.model_validate(stored)
        if signals.source is None:
            signals.source = "analysis_time"
        return signals
    legacy = request_behavior_engine(
        db, _record_to_request(record), record.transaction_id, record.created_at
    )
    return legacy.model_copy(update={"source": "legacy_recomputed"})


def request_behavior_engine(
    db: Session,
    transaction: TransactionAnalyzeRequest,
    transaction_id: str,
    analysis_time: datetime,
) -> BehavioralSignals:
    return BehavioralEngine().calculate(
        db,
        transaction,
        transaction_id=transaction_id,
        analysis_time=analysis_time,
    )


@router.get(
    "/{transaction_id}/network",
    response_model=NetworkSignals,
    summary="Retrieve the analysis-time network signals of a persisted transaction",
    description=(
        "Returns the network result stored when the transaction was analyzed. "
        "Rows analyzed before network persistence existed are recomputed from "
        "history that was already stored at their analysis time and are marked "
        "`source = legacy_recomputed`."
    ),
)
def transaction_network(
    transaction_id: TransactionIdPath,
    db: Session = Depends(get_db),
) -> NetworkSignals:
    record = _load_record_or_404(db, transaction_id)
    stored = record.network_signals
    if isinstance(stored, dict) and stored.get("status") == "available":
        signals = NetworkSignals.model_validate(stored)
        if signals.source is None:
            signals.source = "analysis_time"
        return signals
    legacy = request_network_engine(
        db, _record_to_request(record), record.transaction_id, record.created_at
    )
    return legacy.model_copy(update={"source": "legacy_recomputed"})


def request_network_engine(
    db: Session,
    transaction: TransactionAnalyzeRequest,
    transaction_id: str,
    analysis_time: datetime,
) -> NetworkSignals:
    return NetworkEngine().calculate(
        db,
        transaction,
        transaction_id=transaction_id,
        analysis_time=analysis_time,
    )


@router.get(
    "/{transaction_id}/counterfactual",
    response_model=CounterfactualResponse,
    summary="Compare simple hypothetical amount scenarios",
)
def transaction_counterfactual(
    transaction_id: TransactionIdPath,
    request: Request,
    db: Session = Depends(get_db),
) -> CounterfactualResponse:
    record = _load_record_or_404(db, transaction_id)
    transaction = _record_to_request(record)
    model_loader = getattr(request.app.state, "model_loader", None)
    model_service = getattr(request.app.state, "model_service", None)
    if model_loader is None or model_loader.loaded is None or model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained model services are unavailable",
        )

    engine = getattr(request.app.state, "counterfactual_engine", CounterfactualEngine())
    try:
        return engine.evaluate(
            transaction_id=record.transaction_id,
            transaction=transaction,
            current_probability=record.fraud_probability,
            feature_list=model_loader.loaded.feature_list,
            model_service=model_service,
            risk_level=getattr(record, "risk_level", None),
            behavioral_signals=getattr(record, "behavioral_signals", None),
            network_signals=getattr(record, "network_signals", None),
        )
    except (FeatureEngineeringError, ModelInferenceError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Counterfactual model evaluation is unavailable",
        ) from exc


@router.get(
    "/{transaction_id}",
    response_model=PersistedTransactionResponse,
    summary="Retrieve a persisted transaction analysis",
)
def get_transaction(
    transaction_id: TransactionIdPath,
    db: Session = Depends(get_db),
    actor: Actor = Depends(require_analyst),
) -> PersistedTransactionResponse:
    record = _load_record_or_404(db, transaction_id)
    try:
        review = get_review(db, record.transaction_id)
    except SQLAlchemyError:
        db.rollback()
        review = None
    response = transaction_analysis_to_response(
        record, review_status=review.status if review is not None else None
    )
    response.investigator_summary = build_investigator_summary(_summary_input(response))
    record_audit(
        db,
        actor_or_none(actor),
        "TRANSACTION_VIEW",
        "transaction",
        record.transaction_id,
    )
    return response
