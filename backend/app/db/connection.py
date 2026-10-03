from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


class DatabaseConfigurationError(RuntimeError):
    """Raised when DATABASE_URL is absent."""


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise DatabaseConfigurationError("DATABASE_URL is not configured")
    return database_url


def verify_database_connection() -> dict[str, Any]:
    """Verify PostgreSQL connectivity without returning credentials or secrets."""
    try:
        database_url = get_database_url()
    except DatabaseConfigurationError as exc:
        return {"status": "not_configured", "message": str(exc)}
    psycopg_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    try:
        with psycopg.connect(psycopg_url, connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
    except psycopg.Error as exc:
        return {
            "status": "unavailable",
            "error_type": type(exc).__name__,
            "message": "PostgreSQL connection failed",
        }

    return {"status": "connected", "message": "PostgreSQL connection succeeded"}


if __name__ == "__main__":
    import json

    print(json.dumps(verify_database_connection()))