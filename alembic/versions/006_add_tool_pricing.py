"""Add web_search_call_price_per_thousand to model_pricing

Revision ID: 006
Revises: 005
Create Date: 2026-02-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_pricing",
        sa.Column("web_search_call_price_per_thousand", sa.Numeric(12, 6), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_pricing", "web_search_call_price_per_thousand")
