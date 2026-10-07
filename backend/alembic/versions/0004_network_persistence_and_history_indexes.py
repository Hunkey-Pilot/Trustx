"""persist network signals, request fingerprint and sender/recipient history indexes

Revision ID: 0004_network_persistence
Revises: 0003_behavioral_signals
Create Date: 2026-10-07
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0004_network_persistence"
down_revision: Union[str, Sequence[str], None] = "0003_behavioral_signals"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "transaction_analyses",
        sa.Column("network_signals", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "transaction_analyses",
        sa.Column("request_fingerprint", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_transaction_analyses_request_fingerprint",
        "transaction_analyses",
        ["request_fingerprint"],
    )
    op.create_index(
        "ix_transaction_analyses_name_orig_step",
        "transaction_analyses",
        ["nameOrig", "step"],
    )
    op.create_index(
        "ix_transaction_analyses_name_dest_step",
        "transaction_analyses",
        ["nameDest", "step"],
    )


def downgrade() -> None:
    op.drop_index("ix_transaction_analyses_name_dest_step", table_name="transaction_analyses")
    op.drop_index("ix_transaction_analyses_name_orig_step", table_name="transaction_analyses")
    op.drop_index("ix_transaction_analyses_request_fingerprint", table_name="transaction_analyses")
    op.drop_column("transaction_analyses", "request_fingerprint")
    op.drop_column("transaction_analyses", "network_signals")
