from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

import httpx
from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pricing import ModelPricing

logger = logging.getLogger(__name__)

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

PROVIDER_MAP: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "vertex_ai": "google",
    "gemini": "google",
    "xai": "xai",
    "mistral": "mistral",
}

_PER_MILLION = Decimal("1000000")
_FETCH_TIMEOUT = 30


@dataclass
class SyncResult:
    updated: list[str] = field(default_factory=list)
    unchanged: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "updated": self.updated,
            "updated_count": len(self.updated),
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "errors": self.errors,
        }


def _to_per_million(cost_per_token: float | None) -> Decimal | None:
    if cost_per_token is None or cost_per_token == 0:
        return None
    return (Decimal(str(cost_per_token)) * _PER_MILLION).quantize(Decimal("0.000001"))


def _extract_model_name(key: str) -> str:
    """Strip the provider prefix that LiteLLM uses (e.g. 'xai/grok-2' -> 'grok-2')."""
    if "/" in key:
        return key.split("/", 1)[1]
    return key


def _prices_match(
    existing: ModelPricing,
    input_pm: Decimal,
    output_pm: Decimal,
    image_in: Decimal | None,
    audio_in: Decimal | None,
    audio_out: Decimal | None,
    video_in: Decimal | None,
    web_search_per_1k: Decimal | None,
) -> bool:
    return (
        existing.input_price_per_million == input_pm
        and existing.output_price_per_million == output_pm
        and existing.image_input_price_per_million == image_in
        and existing.audio_input_price_per_million == audio_in
        and existing.audio_output_price_per_million == audio_out
        and existing.video_input_price_per_million == video_in
        and existing.web_search_call_price_per_thousand == web_search_per_1k
    )


def _extract_web_search_price(entry: dict) -> Decimal | None:
    """Extract web search cost per 1000 calls from LiteLLM search_context_cost_per_query.
    LiteLLM uses cost per query (e.g. 0.01 = $0.01/query = $10/1000)."""
    scc = entry.get("search_context_cost_per_query")
    if not isinstance(scc, dict):
        return None
    medium = scc.get("search_context_size_medium")
    low = scc.get("search_context_size_low")
    high = scc.get("search_context_size_high")
    val = medium if medium is not None else (low if low is not None else high)
    if val is None or val <= 0:
        return None
    return (Decimal(str(val)) * 1000).quantize(Decimal("0.01"))


async def fetch_litellm_pricing() -> dict:
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
        resp = await client.get(LITELLM_URL)
        resp.raise_for_status()
        return resp.json()


async def _load_current_prices(db: AsyncSession) -> dict[str, ModelPricing]:
    """Load the latest pricing row for every model currently in the DB."""
    now = datetime.now(timezone.utc)
    latest_sub = (
        select(
            ModelPricing.model_name,
            sa_func.max(ModelPricing.effective_from).label("max_eff"),
        )
        .where(ModelPricing.effective_from <= now)
        .group_by(ModelPricing.model_name)
        .subquery()
    )
    stmt = select(ModelPricing).join(
        latest_sub,
        (ModelPricing.model_name == latest_sub.c.model_name)
        & (ModelPricing.effective_from == latest_sub.c.max_eff),
    )
    result = await db.execute(stmt)
    return {row.model_name: row for row in result.scalars().all()}


async def sync_pricing(db: AsyncSession) -> SyncResult:
    """Fetch LiteLLM pricing data, compare with DB, and insert changed entries."""
    result = SyncResult()

    try:
        raw = await fetch_litellm_pricing()
    except Exception as exc:
        result.errors.append(f"Failed to fetch LiteLLM data: {exc}")
        return result

    current_prices = await _load_current_prices(db)
    now = datetime.now(timezone.utc)
    new_rows: list[ModelPricing] = []

    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue

        litellm_provider = entry.get("litellm_provider", "")
        our_provider = PROVIDER_MAP.get(litellm_provider)
        if our_provider is None:
            result.skipped += 1
            continue

        mode = entry.get("mode", "")
        if mode not in ("chat", "completion"):
            result.skipped += 1
            continue

        input_cpt = entry.get("input_cost_per_token")
        output_cpt = entry.get("output_cost_per_token")
        if input_cpt is None or output_cpt is None:
            result.skipped += 1
            continue

        input_pm = _to_per_million(input_cpt)
        output_pm = _to_per_million(output_cpt)
        if input_pm is None or output_pm is None:
            result.skipped += 1
            continue

        image_in = _to_per_million(entry.get("input_cost_per_image_token"))
        audio_in = _to_per_million(entry.get("input_cost_per_audio_token"))
        audio_out = _to_per_million(entry.get("output_cost_per_audio_token"))
        video_in = _to_per_million(entry.get("input_cost_per_video_token"))
        web_search_per_1k = _extract_web_search_price(entry)

        model_name = _extract_model_name(key)

        existing = current_prices.get(model_name)
        if existing and _prices_match(
            existing, input_pm, output_pm, image_in, audio_in, audio_out, video_in, web_search_per_1k
        ):
            result.unchanged += 1
            continue

        new_rows.append(ModelPricing(
            model_name=model_name,
            provider=our_provider,
            input_price_per_million=input_pm,
            output_price_per_million=output_pm,
            image_input_price_per_million=image_in,
            audio_input_price_per_million=audio_in,
            audio_output_price_per_million=audio_out,
            video_input_price_per_million=video_in,
            web_search_call_price_per_thousand=web_search_per_1k,
            effective_from=now,
        ))
        result.updated.append(model_name)

    if new_rows:
        db.add_all(new_rows)
        await db.commit()
        logger.info("Pricing sync: %d models updated", len(new_rows))
    else:
        logger.info("Pricing sync: all prices up to date")

    return result


# Provider defaults for web search when LiteLLM has no data (per 1000 calls)
_WEB_SEARCH_DEFAULTS: dict[str, Decimal] = {
    "openai": Decimal("10.00"),
    "xai": Decimal("5.00"),
    "google": Decimal("35.00"),
    "mistral": Decimal("10.00"),
}


async def backfill_web_search_pricing(
    db: AsyncSession,
    use_scraper: bool = True,
) -> int:
    """Set web_search_call_price_per_thousand for rows where it is null.
    Uses: (1) scraped vendor pages if use_scraper, (2) provider defaults.
    Returns number of rows updated."""
    from app.services.pricing_scraper import get_scraped_web_search_prices

    scraped: dict[str, Decimal] = {}
    if use_scraper:
        try:
            scraped = await get_scraped_web_search_prices()
        except Exception as exc:
            logger.warning("Pricing scraper failed, using defaults: %s", exc)

    now = datetime.now(timezone.utc)
    latest_sub = (
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
            latest_sub,
            (ModelPricing.model_name == latest_sub.c.model_name)
            & (ModelPricing.effective_from == latest_sub.c.max_eff),
        )
        .where(ModelPricing.web_search_call_price_per_thousand.is_(None))
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    updated = 0
    for row in rows:
        val = scraped.get(row.provider) or _WEB_SEARCH_DEFAULTS.get(row.provider)
        if val is not None:
            row.web_search_call_price_per_thousand = val
            updated += 1
    if updated:
        await db.commit()
        logger.info("Backfill: set web_search_call_price for %d models", updated)
    return updated
