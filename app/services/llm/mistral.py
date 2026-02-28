from __future__ import annotations

import logging
from typing import AsyncGenerator

from mistralai import Mistral

from app.services.llm.base import LLMProvider, TokenUsage

logger = logging.getLogger(__name__)

_EXCLUDED_PREFIXES = ("mistral-embed",)


class MistralProvider(LLMProvider):
    MODELS = ["mistral-large-latest", "mistral-small-latest", "codestral-latest"]

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Mistral API key is not configured")
        self._api_key = api_key
        self._client: Mistral | None = None
        self._live_models: list[str] | None = None

    @property
    def client(self) -> Mistral:
        if self._client is None:
            self._client = Mistral(api_key=self._api_key)
        return self._client

    def provider_name(self) -> str:
        return "mistral"

    async def health_check(self) -> None:
        try:
            await self.client.models.list_async()
        except Exception as exc:
            raise Exception(f"Mistral health check failed: {exc}") from exc

    async def fetch_models(self) -> list[str]:
        try:
            response = await self.client.models.list_async()
            items = getattr(response, "data", None) or []
            models = sorted(
                {
                    m.id
                    for m in items
                    if not any(m.id.startswith(p) for p in _EXCLUDED_PREFIXES)
                }
            )
            if models:
                self._live_models = models
                logger.info("Mistral: fetched %d models", len(models))
        except Exception:
            logger.warning("Mistral: failed to fetch models, using fallback", exc_info=True)
        return self._live_models if self._live_models is not None else list(self.MODELS)

    async def stream_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str | TokenUsage, None]:
        try:
            response = await self.client.chat.stream_async(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = None
            async for chunk in response:
                content = chunk.data.choices[0].delta.content
                if content:
                    yield content
                if getattr(chunk.data, "usage", None):
                    usage = chunk.data.usage
            if usage:
                yield TokenUsage(
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                )
        except Exception as exc:
            if "mistral" in type(exc).__module__.lower():
                raise Exception(f"Mistral API error: {exc}") from exc
            raise

    def supported_models(self) -> list[str]:
        if self._live_models is not None:
            return list(self._live_models)
        return list(self.MODELS)
