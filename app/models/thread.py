from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, String, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.message import Message


class Thread(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "threads"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    llm_name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_deleted: Mapped[bool] = mapped_column(Boolean, server_default="false", nullable=False)

    user: Mapped[User] = relationship("User", back_populates="threads")
    messages: Mapped[list[Message]] = relationship(
        "Message", back_populates="thread", order_by="Message.created_at", lazy="selectin",
    )
