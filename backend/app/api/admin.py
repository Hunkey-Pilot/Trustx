from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.params import Param
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import AuditLog
from app.db.session import get_db
from app.security import require_admin


router = APIRouter(
    prefix="/api/v1/audit-logs",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class AuditLogEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    actor: str
    role: str
    action: str
    resource: str
    resource_id: str | None
    timestamp: datetime
    metadata: dict[str, Any] | None = None


class AuditLogListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    returned_items: int
    items: list[AuditLogEntry]


@router.get("", response_model=AuditLogListResponse, summary="List audit log entries (admin)")
def list_audit_logs(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    action: Annotated[str | None, Query(max_length=64)] = None,
    resource_id: Annotated[str | None, Query(max_length=128)] = None,
) -> AuditLogListResponse:
    action = None if isinstance(action, Param) else action
    resource_id = None if isinstance(resource_id, Param) else resource_id
    query = select(AuditLog)
    if action:
        query = query.where(AuditLog.action == action)
    if resource_id:
        query = query.where(AuditLog.resource_id == resource_id)
    try:
        total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
        rows = db.scalars(
            query.order_by(AuditLog.id.desc()).offset((page - 1) * page_size).limit(page_size)
        ).all()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audit records are temporarily unavailable",
        ) from exc
    items = [
        AuditLogEntry(
            id=row.id,
            actor=row.actor,
            role=row.role,
            action=row.action,
            resource=row.resource,
            resource_id=row.resource_id,
            timestamp=row.timestamp,
            metadata=row.details,
        )
        for row in rows
    ]
    return AuditLogListResponse(
        total=total, page=page, page_size=page_size, returned_items=len(items), items=items
    )
