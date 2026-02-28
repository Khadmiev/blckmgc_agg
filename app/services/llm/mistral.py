from __future__ import annotations

from typing import AsyncGenerator

from mistralai import Mistral

from app.services.llm.base import LLMProvider


class MistralProvider(LLMProvider):
    MODELS = ["mistral-large-latest", "mistral-small-latest", "codestral-latest"]

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Mistral API key is not configured")
        self._api_key = api_key
        self._client: Mistral | None = None

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

    async def stream_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        try:
            response = await self.client.chat.stream_async(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            async for chunk in response:
                content = chunk.data.choices[0].delta.content
                if content:
                    yield content
        except Exception as exc:
            if "mistral" in type(exc).__module__.lower():
                raise Exception(f"Mistral API error: {exc}") from exc
            raise

    def supported_models(self) -> list[str]:
        return list(self.MODELS)
