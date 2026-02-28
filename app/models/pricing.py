from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import DateTime, Index, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKey


class ModelPricing(UUIDPrimaryKey, Base):
    """Append-only pricing ledger. Every price update inserts a new row;
    the current rate for a model is the row with the latest
    ``effective_from <= now()``."""

    __tablename__ = "model_pricing"
    __table_args__ = (
        Index("ix_model_pricing_lookup", "model_name", "effective_from"),
    )

    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)

    input_price_per_million: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False,
    )
    output_price_per_million: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False,
    )

    image_input_price_per_million: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 6), nullable=True,
    )
    audio_input_price_per_million: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 6), nullable=True,
    )
    audio_output_price_per_million: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 6), nullable=True,
    )
    video_input_price_per_million: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 6), nullable=True,
    )

    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
