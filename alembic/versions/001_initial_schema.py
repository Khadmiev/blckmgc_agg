"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-02-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(320), unique=True, nullable=False, index=True),
        sa.Column("username", sa.String(100), nullable=False),
        sa.Column("hashed_password", sa.String(256), nullable=True),
        sa.Column("oauth_provider", sa.String(50), nullable=True),
        sa.Column("oauth_id", sa.String(256), nullable=True),
        sa.Column("avatar_url", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "threads",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("llm_name", sa.String(100), nullable=False),
        sa.Column("is_deleted", sa.Boolean, server_default="false", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("thread_id", UUID(as_uuid=True), sa.ForeignKey("threads.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=True),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("token_count", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "media_attachments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("message_id", UUID(as_uuid=True), sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("media_type", sa.String(20), nullable=False),
        sa.Column("file_path", sa.String(512), nullable=False),
        sa.Column("mime_type", sa.String(100), nullable=False),
        sa.Column("file_size", sa.Integer, nullable=False),
        sa.Column("thumbnail_path", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("media_attachments")
    op.drop_table("messages")
    op.drop_table("threads")
    op.drop_table("users")
