"""Scrape vendor pricing pages for web search / tool call pricing when LiteLLM has no data."""

from __future__ import annotations

import re
import logging
from decimal import Decimal
from typing import NamedTuple

import httpx

logger = logging.getLogger(__name__)

_FETCH_TIMEOUT = 15

# URLs to scrape for web search / tool pricing
_SCRAPE_URLS: dict[str, str] = {
    "openai": "https://platform.openai.com/docs/pricing",
    "anthropic": "https://docs.anthropic.com/en/docs/about-claude/pricing",
    "google": "https://ai.google.dev/gemini-api/docs/pricing",
    "xai": "https://docs.x.ai/docs/guides/live-search",
    "mistral": "https://docs.mistral.ai/agents/connectors/websearch",
}


class ScrapeResult(NamedTuple):
    provider: str
    price_per_thousand: Decimal | None
    source: str
    error: str | None = None


def _parse_price_from_text(text: str, provider: str) -> Decimal | None:
    """Extract web search price per 1000 calls from page text using regex."""
    text_lower = text.lower()
    text_nocomma = text.replace(",", "")

    # Patterns: $X.XX / 1K calls, $X per 1000, etc. (case-insensitive)
    patterns = [
        r"\$(\d+\.?\d*)\s*/\s*1\s*k\s*calls",
        r"\$(\d+\.?\d*)\s*/\s*1,?000\s*calls",
        r"\$(\d+\.?\d*)\s*per\s*1\s*k\s*calls",
        r"\$(\d+\.?\d*)\s*per\s*1,?000\s*(?:calls|queries|requests)",
        r"web\s*search.*?\$(\d+\.?\d*)\s*/\s*1\s*k",
        r"grounding.*?\$(\d+\.?\d*)\s*per\s*1,?000",
        r"\$(\d+\.?\d*)\s*/\s*1,?000\s*search",
        r"\$(\d+\.?\d*)\s*/\s*1k\s*calls",
    ]

    for pat in patterns:
        m = re.search(pat, text_lower, re.DOTALL)
        if m:
            try:
                val = Decimal(m.group(1))
                if 0.1 <= val <= 100:
                    # Sanity: Google grounding is typically $14 or $35
                    if provider == "google" and val not in (Decimal("14"), Decimal("35")):
                        continue
                    return val.quantize(Decimal("0.01"))
            except Exception:
                pass

    # Provider-specific: look for web search section then nearby price
    if provider == "openai":
        if "web search" in text_lower:
            m = re.search(r"\$(\d+\.?\d*)\s*/\s*1\s*k", text_lower)
            if m:
                return Decimal(m.group(1)).quantize(Decimal("0.01"))
            if "10.00" in text_nocomma:
                return Decimal("10.00")
    if provider == "google":
        if "grounding" in text_lower or "search" in text_lower:
            m = re.search(r"\$(\d+)\s*per\s*1,?000\s*(?:search|queries)", text_lower)
            if m:
                v = int(m.group(1))
                if v in (14, 35):
                    return Decimal(str(v)).quantize(Decimal("0.01"))
            if "35" in text and "1000" in text_lower and "grounding" in text_lower:
                return Decimal("35.00")
            if "14" in text and "1000" in text_lower and "grounding" in text_lower:
                return Decimal("14.00")
    if provider == "xai":
        if "5" in text and ("tool" in text_lower or "agent" in text_lower):
            return Decimal("5.00")
    if provider == "mistral":
        if "web search" in text_lower or "websearch" in text_lower:
            if "30" in text and "1000" in text_lower:
                return Decimal("30.00")
            if "10" in text and "1000" in text_lower:
                return Decimal("10.00")

    return None


async def scrape_web_search_pricing() -> dict[str, ScrapeResult]:
    """Fetch vendor pricing pages and extract web search / tool pricing.
    Returns {provider: ScrapeResult}."""
    results: dict[str, ScrapeResult] = {}

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LPO-Pricing-Scraper/1.0; +https://github.com)",
        "Accept": "text/html,application/xhtml+xml",
    }
    async with httpx.AsyncClient(
        timeout=_FETCH_TIMEOUT, follow_redirects=True, headers=headers
    ) as client:
        for provider, url in _SCRAPE_URLS.items():
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                text = resp.text
                price = _parse_price_from_text(text, provider)
                results[provider] = ScrapeResult(
                    provider=provider,
                    price_per_thousand=price,
                    source=url,
                )
                if price:
                    logger.info("Scraper %s: %s per 1000 calls", provider, price)
                else:
                    logger.debug("Scraper %s: no price found", provider)
            except Exception as exc:
                logger.warning("Scraper %s failed: %s", provider, exc)
                results[provider] = ScrapeResult(
                    provider=provider,
                    price_per_thousand=None,
                    source=url,
                    error=str(exc),
                )

    return results


async def get_scraped_web_search_prices() -> dict[str, Decimal]:
    """Scrape vendor pages and return {provider: price} for providers where we got a value."""
    results = await scrape_web_search_pricing()
    return {
        r.provider: r.price_per_thousand
        for r in results.values()
        if r.price_per_thousand is not None
    }
