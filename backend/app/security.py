"""Minimal API-key authentication for the TrustX hackathon API.

Two roles are supported, ANALYST and ADMIN (ADMIN includes ANALYST rights).
Keys come only from environment variables and are compared in constant time:

    TRUSTX_ANALYST_API_KEY
    TRUSTX_ADMIN_API_KEY

If a role's key is not configured, that role cannot authenticate (fail closed).
The optional ``X-TrustX-User`` header lets an analyst label their own actions
for the audit log / review record. It is self-asserted display text, not an
authenticated identity.
"""

from __future__ import annotations

import hmac
import os
import re
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Header, HTTPException, status


ROLE_ANALYST = "ANALYST"
ROLE_ADMIN = "ADMIN"

_USER_LABEL = re.compile(r"[^A-Za-z0-9_.@-]")


@dataclass(frozen=True)
class Actor:
    name: str
    role: str


def _configured_keys() -> dict[str, str]:
    keys = {
        ROLE_ADMIN: os.getenv("TRUSTX_ADMIN_API_KEY", ""),
        ROLE_ANALYST: os.getenv("TRUSTX_ANALYST_API_KEY", ""),
    }
    return {role: key for role, key in keys.items() if key}


def _sanitize_user_label(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = _USER_LABEL.sub("", value.strip())[:48]
    return cleaned or None


def authenticate(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_trustx_user: str | None = Header(default=None, alias="X-TrustX-User"),
) -> Actor:
    keys = _configured_keys()
    if not keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured on the server",
        )
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    matched_role: str | None = None
    for role in (ROLE_ADMIN, ROLE_ANALYST):
        key = keys.get(role)
        if key and hmac.compare_digest(x_api_key.encode("utf-8"), key.encode("utf-8")):
            matched_role = role
            break
    if matched_role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    label = _sanitize_user_label(x_trustx_user)
    name = matched_role.lower() + (f":{label}" if label else "")
    return Actor(name=name, role=matched_role)


def require_analyst(actor: Actor = Depends(authenticate)) -> Actor:
    """ANALYST or ADMIN."""
    return actor


def require_admin(actor: Actor = Depends(authenticate)) -> Actor:
    if actor.role != ROLE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator role required",
        )
    return actor


def actor_or_none(value: Any) -> Actor | None:
    """Return an Actor, or None when a route function is called directly in tests."""
    return value if isinstance(value, Actor) else None
