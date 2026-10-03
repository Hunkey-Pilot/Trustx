"""add transaction history filter indexes

Revision ID: 0002_history_indexes
Revises: 0001_create_transaction_analyses
Create Date: 2026-10-04
"""

from typing import Sequence, Union

from alembic import op


revision: str = "0002_history_indexes"
down_revision: Union[str, Sequence[str], None] = "0001_create_transaction_analyses"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_transaction_analyses_type", "transaction_analyses", ["type"])
    op.create_index(
        "ix_transaction_analyses_recommended_action",
        "transaction_analyses",
        ["recommended_action"],
    )
    op.create_index(
        "ix_transaction_analyses_fraud_probability",
        "transaction_analyses",
        ["fraud_probability"],
    )
    op.create_index("ix_transaction_analyses_risk_score", "transaction_analyses", ["risk_score"])


def downgrade() -> None:
    op.drop_index("ix_transaction_analyses_risk_score", table_name="transaction_analyses")
    op.drop_index("ix_transaction_analyses_fraud_probability", table_name="transaction_analyses")
    op.drop_index("ix_transaction_analyses_recommended_action", table_name="transaction_analyses")
    op.drop_index("ix_transaction_analyses_type", table_name="transaction_analyses")