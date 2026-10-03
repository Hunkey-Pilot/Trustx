from __future__ import annotations

from uuid import uuid4

import pandas as pd
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.params import Param
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.schemas.transactions import (
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
from app.db.repository import (
    create_transaction_analysis,
    get_transaction_analysis,
    list_transaction_analyses,
    transaction_summary,
    transaction_analysis_to_response,
)
from app.db.session import get_db
from app.services.feature_engineering import (
    FeatureEngineeringError,
    create_model_features,
)
from app.services.evidence_engine import EvidenceEngine
from app.services.behavioral_engine import BehavioralEngine
from app.services.network_engine import NetworkEngine
from app.services.counterfactual_engine import CounterfactualEngine
from app.services.model_service import ModelInferenceError
from app.services.risk_engine import RiskEngineError
from app.services.shap_service import ShapExplanation as ShapServiceExplanation


router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RecommendedAction = Literal[
    "ALLOW / MONITOR",
    "ADDITIONAL_VERIFICATION / MONITOR",
    "STEP_UP_VERIFICATION / MANUAL_REVIEW",
    "TEMPORARY_HOLD / ESCALATE_FOR_REVIEW",
]
SortField = Literal["created_at", "fraud_probability", "risk_score", "amount"]
SortOrder = Literal["asc", "desc"]


@router.post(
    "/analyze",
    response_model=TransactionAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze one transaction with the trained models",
    description=(
        "Builds the exact 28-feature model vector from the transaction and "
        "historical sender context, then returns the trained XGBoost probability "
        "and raw Isolation Forest anomaly signal."
    ),
)
def analyze_transaction(
    request: Request,
    transaction: TransactionAnalyzeRequest,
    db: Session = Depends(get_db),
) -> TransactionAnalyzeResponse:
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
    except (FeatureEngineeringError, ModelInferenceError, RiskEngineError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    risk_result = request.app.state.risk_engine.calculate(
        fraud_probability=prediction.fraud_probability,
        anomaly_signal=prediction.anomaly_signal,
    )
    shap_result: ShapServiceExplanation = request.app.state.shap_service.explain(features)
    evidence_result = request.app.state.evidence_engine.aggregate(
        fraud_probability=prediction.fraud_probability,
        anomaly_signal=prediction.anomaly_signal,
        shap_explanation=shap_result,
        risk_result=risk_result,
    )
    transaction_id = f"req_{uuid4().hex}"
    behavioral_engine = getattr(request.app.state, "behavioral_engine", None)
    network_engine = getattr(request.app.state, "network_engine", None)
    if isinstance(db, Session) and behavioral_engine is not None:
        behavioral_signals = behavioral_engine.calculate(
            db,
            transaction,
            transaction_id=transaction_id,
            analysis_time=datetime.now(timezone.utc),
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
            analysis_time=datetime.now(timezone.utc),
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
            error=shap_result.error,
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

    if isinstance(db, Session):
        try:
            create_transaction_analysis(db, transaction, response)
        except SQLAlchemyError as exc:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Analysis completed but could not be persisted",
            ) from exc

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
    transaction_id: str | None = Query(None, min_length=1),
    created_after: datetime | None = Query(None),
    created_before: datetime | None = Query(None),
    min_fraud_probability: float | None = Query(None, ge=0, le=1),
    max_fraud_probability: float | None = Query(None, ge=0, le=1),
    min_risk_score: float | None = Query(None, ge=0, le=100),
    max_risk_score: float | None = Query(None, ge=0, le=100),
    sort_by: SortField = Query("created_at"),
    sort_order: SortOrder = Query("desc"),
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
    transaction_id: str,
    db: Session = Depends(get_db),
) -> BehavioralSignals:
    try:
        record = get_transaction_analysis(db, transaction_id)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction record is temporarily unavailable",
        ) from exc
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    transaction = TransactionAnalyzeRequest.model_validate(
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
    return request_behavior_engine(db, transaction, record.transaction_id, record.created_at)


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
    summary="Retrieve network signals for a persisted transaction",
)
def transaction_network(
    transaction_id: str,
    db: Session = Depends(get_db),
) -> NetworkSignals:
    try:
        record = get_transaction_analysis(db, transaction_id)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction record is temporarily unavailable",
        ) from exc
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    transaction = TransactionAnalyzeRequest.model_validate(
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
    return request_network_engine(db, transaction, record.transaction_id, record.created_at)


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
    transaction_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> CounterfactualResponse:
    try:
        record = get_transaction_analysis(db, transaction_id)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction record is temporarily unavailable",
        ) from exc
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")

    transaction = TransactionAnalyzeRequest.model_validate(
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
    transaction_id: str,
    db: Session = Depends(get_db),
) -> PersistedTransactionResponse:
    try:
        record = get_transaction_analysis(db, transaction_id)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction record is temporarily unavailable",
        ) from exc
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    return transaction_analysis_to_response(record)