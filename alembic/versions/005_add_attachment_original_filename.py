"""Add original_filename to media_attachments

Revision ID: 005
Revises: 004
"""
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column(
        "media_attachments",
        sa.Column("original_filename", sa.String(512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("media_attachments", "original_filename")
