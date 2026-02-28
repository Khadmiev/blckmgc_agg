"""Add prompt_tokens and completion_tokens to messages

Revision ID: 002
Revises: 001
Create Date: 2026-02-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("prompt_tokens", sa.Integer, nullable=True))
    op.add_column("messages", sa.Column("completion_tokens", sa.Integer, nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "completion_tokens")
    op.drop_column("messages", "prompt_tokens")
