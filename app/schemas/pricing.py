from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PricingCreate(BaseModel):
    model_name: str = Field(..., max_length=100)
    provider: str = Field(..., max_length=50)

    input_price_per_million: float
    output_price_per_million: float

    image_input_price_per_million: Optional[float] = None
    audio_input_price_per_million: Optional[float] = None
    audio_output_price_per_million: Optional[float] = None
    video_input_price_per_million: Optional[float] = None

    effective_from: Optional[datetime] = None


class PricingBulkCreate(BaseModel):
    items: list[PricingCreate]


class PricingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    model_name: str
    provider: str

    input_price_per_million: float
    output_price_per_million: float

    image_input_price_per_million: Optional[float] = None
    audio_input_price_per_million: Optional[float] = None
    audio_output_price_per_million: Optional[float] = None
    video_input_price_per_million: Optional[float] = None

    effective_from: datetime
    created_at: datetime
