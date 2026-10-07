from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import AuditLog
from app.security import Actor


logger = logging.getLogger("trustx.audit")

_FORBIDDEN_KEYS = {"api_key", "x-api-key", "password", "secret", "token", "authorization"}


def _clean_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata:
        return None
    return {
        key: value
        for key, value in metadata.items()
        if key.lower() not in _FORBIDDEN_KEYS
    }


def record_audit(
    db: Any,
    actor: Actor | None,
    action: str,
    resource: str,
    resource_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist an audit entry. Never raises: auditing must not break a request,
    but a failure is logged (without request data or credentials)."""
    if actor is None or not isinstance(db, Session):
        return
    try:
        db.add(
            AuditLog(
                actor=actor.name,
                role=actor.role,
                action=action,
                resource=resource,
                resource_id=resource_id,
                details=_clean_metadata(metadata),
            )
        )
        db.commit()
    except Exception:  # noqa: BLE001 - audit failures must not break the request
        db.rollback()
        logger.error("Failed to write audit log entry for action %s", action)
