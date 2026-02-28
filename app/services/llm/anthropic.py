from __future__ import annotations

import logging
from typing import AsyncGenerator

from anthropic import AnthropicError, AsyncAnthropic

from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    MODELS = ["claude-sonnet-4-20250514", "claude-haiku-4-20250414"]

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Anthropic API key is not configured")
        self._api_key = api_key
        self._client: AsyncAnthropic | None = None
        self._live_models: list[str] | None = None

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    def provider_name(self) -> str:
        return "anthropic"

    async def health_check(self) -> None:
        try:
            await self.client.messages.create(
                model=self.supported_models()[0],
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
        except AnthropicError as exc:
            raise Exception(f"Anthropic health check failed: {exc}") from exc

    async def fetch_models(self) -> list[str]:
        try:
            models: list[str] = []
            async for page in self.client.models.list():
                if hasattr(page, "id") and page.id.startswith("claude-"):
                    models.append(page.id)
            if models:
                self._live_models = sorted(models)
                logger.info("Anthropic: fetched %d models", len(models))
        except Exception:
            logger.warning("Anthropic: failed to fetch models, using fallback", exc_info=True)
        return self._live_models if self._live_models is not None else list(self.MODELS)

    async def stream_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        try:
            system: str | None = None
            filtered: list[dict] = []
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    filtered.append(msg)

            kwargs: dict = {
                "model": model,
                "messages": filtered,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if system:
                kwargs["system"] = system

            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except AnthropicError as exc:
            raise Exception(f"Anthropic API error: {exc}") from exc

    def supported_models(self) -> list[str]:
        if self._live_models is not None:
            return list(self._live_models)
        return list(self.MODELS)
