from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_db
from app.models.pricing import ModelPricing
from app.schemas.pricing import PricingBulkCreate, PricingCreate, PricingResponse
from app.services.pricing_sync import sync_pricing

router = APIRouter()


async def _require_pricing_key(
    x_pricing_api_key: str = Header(..., alias="X-Pricing-Api-Key"),
) -> None:
    if not settings.pricing_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pricing administration is not configured",
        )
    if x_pricing_api_key != settings.pricing_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid pricing API key",
        )


def _row_from_body(body: PricingCreate) -> ModelPricing:
    return ModelPricing(
        model_name=body.model_name,
        provider=body.provider,
        input_price_per_million=body.input_price_per_million,
        output_price_per_million=body.output_price_per_million,
        image_input_price_per_million=body.image_input_price_per_million,
        audio_input_price_per_million=body.audio_input_price_per_million,
        audio_output_price_per_million=body.audio_output_price_per_million,
        video_input_price_per_million=body.video_input_price_per_million,
        effective_from=body.effective_from or datetime.now(timezone.utc),
    )


@router.post("/", response_model=PricingResponse, status_code=status.HTTP_201_CREATED)
async def create_pricing(
    body: PricingCreate,
    _key: None = Depends(_require_pricing_key),
    db: AsyncSession = Depends(get_db),
):
    row = _row_from_body(body)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


@router.post("/bulk", response_model=list[PricingResponse], status_code=status.HTTP_201_CREATED)
async def create_pricing_bulk(
    body: PricingBulkCreate,
    _key: None = Depends(_require_pricing_key),
    db: AsyncSession = Depends(get_db),
):
    rows = [_row_from_body(item) for item in body.items]
    db.add_all(rows)
    await db.commit()
    for row in rows:
        await db.refresh(row)
    return rows


@router.get("/", response_model=list[PricingResponse])
async def list_current_pricing(
    _key: None = Depends(_require_pricing_key),
    db: AsyncSession = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    latest_per_model = (
        select(
            ModelPricing.model_name,
            sa_func.max(ModelPricing.effective_from).label("max_eff"),
        )
        .where(ModelPricing.effective_from <= now)
        .group_by(ModelPricing.model_name)
        .subquery()
    )
    stmt = (
        select(ModelPricing)
        .join(
            latest_per_model,
            (ModelPricing.model_name == latest_per_model.c.model_name)
            & (ModelPricing.effective_from == latest_per_model.c.max_eff),
        )
        .order_by(ModelPricing.provider, ModelPricing.model_name)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("/sync")
async def sync_pricing_from_litellm(
    _key: None = Depends(_require_pricing_key),
    db: AsyncSession = Depends(get_db),
):
    result = await sync_pricing(db)
    return result.to_dict()


@router.get("/history", response_model=list[PricingResponse])
async def pricing_history(
    model_name: Optional[str] = Query(None),
    _key: None = Depends(_require_pricing_key),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ModelPricing).order_by(
        ModelPricing.model_name, ModelPricing.effective_from.desc(),
    )
    if model_name:
        stmt = stmt.where(ModelPricing.model_name == model_name)
    result = await db.execute(stmt)
    return result.scalars().all()
