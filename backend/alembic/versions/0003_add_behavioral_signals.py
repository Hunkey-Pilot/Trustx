"""add behavioral signals to transaction analyses

Revision ID: 0003_behavioral_signals
Revises: 0002_history_indexes
Create Date: 2026-10-04
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0003_behavioral_signals"
down_revision: Union[str, Sequence[str], None] = "0002_history_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "transaction_analyses",
        sa.Column("behavioral_signals", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("transaction_analyses", "behavioral_signals")