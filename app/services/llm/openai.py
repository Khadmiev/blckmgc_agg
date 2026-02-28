from __future__ import annotations

from typing import AsyncGenerator

from openai import AsyncOpenAI, OpenAIError

from app.services.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o3-mini"]

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is not configured")
        self._api_key = api_key
        self._client: AsyncOpenAI | None = None

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
        return list(self.MODELS)
