from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey

if TYPE_CHECKING:
    from app.models.thread import Thread


class User(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), nullable=False)
    hashed_password: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    oauth_provider: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    oauth_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    threads: Mapped[list[Thread]] = relationship("Thread", back_populates="user", lazy="selectin")
