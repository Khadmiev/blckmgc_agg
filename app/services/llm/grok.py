from __future__ import annotations

import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI, OpenAIError

from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1"


class GrokProvider(LLMProvider):
    MODELS = ["grok-3", "grok-3-mini"]

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("xAI API key is not configured")
        self._api_key = api_key
        self._client: AsyncOpenAI | None = None
        self._live_models: list[str] | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=XAI_BASE_URL)
        return self._client

    def provider_name(self) -> str:
        return "xai"

    async def health_check(self) -> None:
        try:
            await self.client.models.list()
        except OpenAIError as exc:
            raise Exception(f"xAI/Grok health check failed: {exc}") from exc

    async def fetch_models(self) -> list[str]:
        try:
            response = await self.client.models.list()
            models = sorted(
                {m.id for m in response.data if m.id.startswith("grok-")}
            )
            if models:
                self._live_models = models
                logger.info("xAI/Grok: fetched %d models", len(models))
        except Exception:
            logger.warning("xAI/Grok: failed to fetch models, using fallback", exc_info=True)
        return self._live_models if self._live_models is not None else list(self.MODELS)

    async def stream_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except OpenAIError as exc:
            raise Exception(f"xAI/Grok API error: {exc}") from exc

    def supported_models(self) -> list[str]:
        if self._live_models is not None:
            return list(self._live_models)
        return list(self.MODELS)
