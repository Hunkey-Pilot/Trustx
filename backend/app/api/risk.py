from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.repository import transaction_summary
from app.db.session import get_db
from app.schemas.transactions import TransactionSummaryResponse
from app.security import require_analyst


router = APIRouter(
    prefix="/api/v1/risk",
    tags=["risk"],
    dependencies=[Depends(require_analyst)],
)


@router.get("/summary", response_model=TransactionSummaryResponse)
def risk_summary(db: Session = Depends(get_db)) -> TransactionSummaryResponse:
    try:
        return transaction_summary(db)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk summary is temporarily unavailable",
        ) from exc