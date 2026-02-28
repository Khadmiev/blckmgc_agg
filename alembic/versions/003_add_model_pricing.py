"""Add model_pricing table and messages.cost_usd column

Revision ID: 003
Revises: 002
Create Date: 2026-02-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_pricing",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("input_price_per_million", sa.Numeric(12, 6), nullable=False),
        sa.Column("output_price_per_million", sa.Numeric(12, 6), nullable=False),
        sa.Column("image_input_price_per_million", sa.Numeric(12, 6), nullable=True),
        sa.Column("audio_input_price_per_million", sa.Numeric(12, 6), nullable=True),
        sa.Column("audio_output_price_per_million", sa.Numeric(12, 6), nullable=True),
        sa.Column("video_input_price_per_million", sa.Numeric(12, 6), nullable=True),
        sa.Column("effective_from", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_model_pricing_lookup",
        "model_pricing",
        ["model_name", "effective_from"],
    )

    op.add_column("messages", sa.Column("cost_usd", sa.Numeric(10, 6), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "cost_usd")
    op.drop_index("ix_model_pricing_lookup", table_name="model_pricing")
    op.drop_table("model_pricing")
