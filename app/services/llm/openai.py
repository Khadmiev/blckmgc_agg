from __future__ import annotations

import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI, OpenAIError

from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_CHAT_PREFIXES = ("gpt-3.5-turbo", "gpt-4", "o1", "o3", "o4", "chatgpt")
_EXCLUDED_KEYWORDS = ("realtime", "audio", "search")


class OpenAIProvider(LLMProvider):
    MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o3-mini"]

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is not configured")
        self._api_key = api_key
        self._client: AsyncOpenAI | None = None
        self._live_models: list[str] | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    def provider_name(self) -> str:
        return "openai"

    async def health_check(self) -> None:
        try:
            await self.client.models.list()
        except OpenAIError as exc:
            raise Exception(f"OpenAI health check failed: {exc}") from exc

    @staticmethod
    def _is_chat_model(model_id: str) -> bool:
        if any(kw in model_id for kw in _EXCLUDED_KEYWORDS):
            return False
        return any(model_id.startswith(p) for p in _CHAT_PREFIXES)

    async def fetch_models(self) -> list[str]:
        try:
            response = await self.client.models.list()
            models = sorted(
                {m.id for m in response.data if self._is_chat_model(m.id)}
            )
            if models:
                self._live_models = models
                logger.info("OpenAI: fetched %d chat models", len(models))
        except Exception:
            logger.warning("OpenAI: failed to fetch models, using fallback", exc_info=True)
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
            raise Exception(f"OpenAI API error: {exc}") from exc

    def supported_models(self) -> list[str]:
        if self._live_models is not None:
            return list(self._live_models)
        return list(self.MODELS)
