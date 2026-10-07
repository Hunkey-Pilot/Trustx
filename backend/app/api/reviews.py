from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.params import Param
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import CaseReview
from app.db.repository import (
    get_review,
    get_transaction_analysis,
    list_reviews,
    review_to_response,
)
from app.db.session import get_db
from app.schemas.transactions import (
    MAX_TRANSACTION_ID_LENGTH,
    TRANSACTION_ID_PATTERN,
    ReviewCreateRequest,
    ReviewListResponse,
    ReviewResponse,
    ReviewStatus,
    ReviewUpdateRequest,
)
from app.security import Actor, actor_or_none, require_analyst
from app.services.audit_service import record_audit


router = APIRouter(
    prefix="/api/v1/reviews",
    tags=["reviews"],
    dependencies=[Depends(require_analyst)],
)

TransactionIdPath = Annotated[
    str,
    Path(min_length=1, max_length=MAX_TRANSACTION_ID_LENGTH, pattern=TRANSACTION_ID_PATTERN),
]
RiskLevelFilter = Annotated[str, Query(pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")]


def _unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Review records are temporarily unavailable",
    )


def _default(value, default):
    return default if isinstance(value, Param) else value


@router.get("", response_model=ReviewListResponse, summary="List analyst reviews")
def list_all_reviews(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    review_status: ReviewStatus | None = Query(None, alias="status"),
    risk_level: RiskLevelFilter | None = None,
) -> ReviewListResponse:
    try:
        return list_reviews(
            db,
            page=page,
            page_size=page_size,
            status=_default(review_status, None),
            risk_level=_default(risk_level, None),
        )
    except SQLAlchemyError as exc:
        raise _unavailable() from exc


@router.get(
    "/{transaction_id}",
    response_model=ReviewResponse,
    summary="Get the analyst review of a transaction",
)
def get_transaction_review(
    transaction_id: TransactionIdPath,
    db: Session = Depends(get_db),
) -> ReviewResponse:
    try:
        review = get_review(db, transaction_id)
        analysis = get_transaction_analysis(db, transaction_id)
    except SQLAlchemyError as exc:
        raise _unavailable() from exc
    if review is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
    return review_to_response(review, analysis)


@router.post(
    "/{transaction_id}",
    response_model=ReviewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an analyst review for a transaction",
)
def create_review(
    transaction_id: TransactionIdPath,
    payload: ReviewCreateRequest,
    db: Session = Depends(get_db),
    actor: Actor = Depends(require_analyst),
) -> ReviewResponse:
    actor_value = actor_or_none(actor)
    try:
        analysis = get_transaction_analysis(db, transaction_id)
        if analysis is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
        if get_review(db, transaction_id) is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A review already exists for this transaction; use PATCH to update it",
            )
        now = datetime.now(timezone.utc)
        review = CaseReview(
            transaction_id=transaction_id,
            status=payload.status,
            analyst_note=payload.analyst_note,
            decision=payload.decision,
            reviewed_by=actor_value.name if actor_value else None,
            reviewed_at=now,
            created_at=now,
            updated_at=now,
        )
        db.add(review)
        db.commit()
        db.refresh(review)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A review already exists for this transaction; use PATCH to update it",
        ) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise _unavailable() from exc
    record_audit(
        db,
        actor_value,
        "REVIEW_CREATE",
        "review",
        transaction_id,
        {"status": review.status, "has_note": bool(review.analyst_note)},
    )
    return review_to_response(review, analysis)


@router.patch(
    "/{transaction_id}",
    response_model=ReviewResponse,
    summary="Update an analyst review",
)
def update_review(
    transaction_id: TransactionIdPath,
    payload: ReviewUpdateRequest,
    db: Session = Depends(get_db),
    actor: Actor = Depends(require_analyst),
) -> ReviewResponse:
    actor_value = actor_or_none(actor)
    fields = payload.model_fields_set
    if not fields:
        raise HTTPException(status_code=422, detail="Provide at least one field to update")
    if "status" in fields and payload.status is None:
        raise HTTPException(status_code=422, detail="status cannot be null")
    try:
        review = get_review(db, transaction_id)
        if review is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
        previous_status = review.status
        if "status" in fields:
            review.status = payload.status
        if "analyst_note" in fields:
            review.analyst_note = payload.analyst_note
        if "decision" in fields:
            review.decision = payload.decision
        now = datetime.now(timezone.utc)
        review.reviewed_by = actor_value.name if actor_value else review.reviewed_by
        review.reviewed_at = now
        review.updated_at = now
        db.commit()
        db.refresh(review)
        analysis = get_transaction_analysis(db, transaction_id)
    except SQLAlchemyError as exc:
        db.rollback()
        raise _unavailable() from exc
    record_audit(
        db,
        actor_value,
        "REVIEW_UPDATE",
        "review",
        transaction_id,
        {
            "previous_status": previous_status,
            "status": review.status,
            "note_changed": "analyst_note" in fields,
        },
    )
    return review_to_response(review, analysis)
