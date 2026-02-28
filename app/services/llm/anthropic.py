from __future__ import annotations

from typing import AsyncGenerator

from anthropic import AnthropicError, AsyncAnthropic

from app.services.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    MODELS = ["claude-sonnet-4-20250514", "claude-haiku-4-20250414"]

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Anthropic API key is not configured")
        self._api_key = api_key
        self._client: AsyncAnthropic | None = None

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
                model=self.MODELS[0],
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
        except AnthropicError as exc:
            raise Exception(f"Anthropic health check failed: {exc}") from exc

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
        return list(self.MODELS)
