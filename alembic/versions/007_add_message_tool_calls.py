"""Add web_search_calls and tool_calls to messages

Revision ID: 007
Revises: 006
Create Date: 2026-02-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("web_search_calls", sa.Integer(), nullable=True),
    )
    op.add_column(
        "messages",
        sa.Column("tool_calls", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("messages", "tool_calls")
    op.drop_column("messages", "web_search_calls")
