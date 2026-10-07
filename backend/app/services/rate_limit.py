"""Tiny in-process sliding-window rate limiter for the analyze endpoint.

This is intentionally minimal (single process, in memory). It protects the
expensive scoring path from accidental floods during a demo; it is not a
substitute for gateway-level rate limiting in a real deployment.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, status

from app.security import Actor, require_analyst


WINDOW_SECONDS = 60.0

_lock = threading.Lock()
_hits: dict[str, deque[float]] = defaultdict(deque)


def _limit_per_minute() -> int:
    try:
        return int(os.getenv("TRUSTX_RATE_LIMIT_PER_MINUTE", "120"))
    except ValueError:
        return 120


def reset_rate_limiter() -> None:
    with _lock:
        _hits.clear()


def enforce_analyze_rate_limit(actor: Actor = Depends(require_analyst)) -> Actor:
    limit = _limit_per_minute()
    if limit <= 0:
        return actor
    now = time.monotonic()
    with _lock:
        window = _hits[actor.role]
        while window and now - window[0] > WINDOW_SECONDS:
            window.popleft()
        if len(window) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded; retry shortly",
                headers={"Retry-After": "10"},
            )
        window.append(now)
    return actor
