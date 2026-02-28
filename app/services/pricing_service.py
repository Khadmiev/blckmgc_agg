from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pricing import ModelPricing
from app.services.llm.base import TokenUsage

TOKENS_PER_IMAGE = 1000
TOKENS_PER_AUDIO_SECOND = 25
TOKENS_PER_VIDEO_SECOND = 50

_ONE_MILLION = Decimal("1000000")


@dataclass(frozen=True, slots=True)
class MediaCounts:
    image_count: int = 0
    audio_seconds: float = 0.0
    video_seconds: float = 0.0

    @property
    def estimated_image_tokens(self) -> int:
        return self.image_count * TOKENS_PER_IMAGE

    @property
    def estimated_audio_tokens(self) -> int:
        return int(self.audio_seconds * TOKENS_PER_AUDIO_SECOND)

    @property
    def estimated_video_tokens(self) -> int:
        return int(self.video_seconds * TOKENS_PER_VIDEO_SECOND)

    @property
    def total_media_tokens(self) -> int:
        return (
            self.estimated_image_tokens
            + self.estimated_audio_tokens
            + self.estimated_video_tokens
        )


async def get_current_price(
    db: AsyncSession, model_name: str,
) -> ModelPricing | None:
    now = datetime.now(timezone.utc)
    stmt = (
        select(ModelPricing)
        .where(
            ModelPricing.model_name == model_name,
            ModelPricing.effective_from <= now,
        )
        .order_by(ModelPricing.effective_from.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


def compute_cost(
    usage: TokenUsage,
    pricing: ModelPricing,
    media: MediaCounts | None = None,
) -> Decimal:
    if media is None:
        media = MediaCounts()

    image_tokens = media.estimated_image_tokens
    audio_in_tokens = media.estimated_audio_tokens
    video_tokens = media.estimated_video_tokens
    total_media = media.total_media_tokens

    text_input = max(usage.prompt_tokens - total_media, 0)

    input_rate = pricing.input_price_per_million
    output_rate = pricing.output_price_per_million
    image_rate = pricing.image_input_price_per_million or input_rate
    audio_in_rate = pricing.audio_input_price_per_million or input_rate
    audio_out_rate = pricing.audio_output_price_per_million or output_rate
    video_rate = pricing.video_input_price_per_million or input_rate

    cost = (
        Decimal(text_input) * input_rate
        + Decimal(image_tokens) * image_rate
        + Decimal(audio_in_tokens) * audio_in_rate
        + Decimal(video_tokens) * video_rate
        + Decimal(usage.completion_tokens) * output_rate
    ) / _ONE_MILLION

    return cost.quantize(Decimal("0.000001"))
