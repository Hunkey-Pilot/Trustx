"""create transaction analyses table

Revision ID: 0001_create_transaction_analyses
Revises:
Create Date: 2026-10-04
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0001_create_transaction_analyses"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "transaction_analyses",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("transaction_id", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=20), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("nameOrig", sa.String(length=128), nullable=False),
        sa.Column("oldbalanceOrg", sa.Float(), nullable=False),
        sa.Column("newbalanceOrig", sa.Float(), nullable=False),
        sa.Column("nameDest", sa.String(length=128), nullable=False),
        sa.Column("oldbalanceDest", sa.Float(), nullable=False),
        sa.Column("newbalanceDest", sa.Float(), nullable=False),
        sa.Column("historical_transactions", postgresql.JSONB(), nullable=False),
        sa.Column("fraud_probability", sa.Float(), nullable=False),
        sa.Column("anomaly_signal", sa.Float(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=20), nullable=False),
        sa.Column("recommended_action", sa.Text(), nullable=False),
        sa.Column("explanation", postgresql.JSONB(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=True),
        sa.Column("feature_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("transaction_id"),
    )
    op.create_index("ix_transaction_analyses_transaction_id", "transaction_analyses", ["transaction_id"])
    op.create_index("ix_transaction_analyses_created_at", "transaction_analyses", ["created_at"])
    op.create_index("ix_transaction_analyses_risk_level", "transaction_analyses", ["risk_level"])


def downgrade() -> None:
    op.drop_index("ix_transaction_analyses_risk_level", table_name="transaction_analyses")
    op.drop_index("ix_transaction_analyses_created_at", table_name="transaction_analyses")
    op.drop_index("ix_transaction_analyses_transaction_id", table_name="transaction_analyses")
    op.drop_table("transaction_analyses")