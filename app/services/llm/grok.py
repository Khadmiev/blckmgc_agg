from __future__ import annotations

import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI, OpenAIError

from app.config import settings
from app.services.llm.base import LLMProvider, TokenUsage

logger = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1"

_RESPONSE_TOOLS = [{"type": "web_search"}]


def _messages_to_input(messages: list[dict]) -> list[dict]:
    """Convert chat messages to Responses API input format."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


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
    ) -> AsyncGenerator[str | TokenUsage, None]:
        try:
            if settings.use_response_apis:
                async for item in self._stream_responses_api(messages, model, temperature, max_tokens):
                    yield item
            else:
                async for item in self._stream_chat_completions(messages, model, temperature, max_tokens):
                    yield item
        except OpenAIError as exc:
            raise Exception(f"xAI/Grok API error: {exc}") from exc

    async def _stream_responses_api(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str | TokenUsage, None]:
        """Use Responses API with web search and other agent tools."""
        input_items = _messages_to_input(messages)
        try:
            response = await self.client.responses.create(
                model=model,
                input=input_items,
                tools=_RESPONSE_TOOLS,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        except (AttributeError, TypeError, OpenAIError) as e:
            logger.warning("xAI Responses API not available, falling back to Chat: %s", e)
            async for item in self._stream_chat_completions(messages, model, temperature, max_tokens):
                yield item
            return
        text = ""
        if hasattr(response, "output") and response.output:
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for part in item.content:
                        if hasattr(part, "text") and part.text:
                            text += part.text
        if hasattr(response, "output_text"):
            text = response.output_text or text
        if text:
            yield text
        usage = getattr(response, "usage", None)
        tool_calls = 0
        web_search_calls = 0
        sstu = getattr(response, "server_side_tool_usage", None)
        if sstu and isinstance(sstu, dict):
            for _key, count in sstu.items():
                n = count if isinstance(count, int) else 0
                tool_calls += n
                if "WEB_SEARCH" in str(_key).upper():
                    web_search_calls += n
        if web_search_calls == 0 and tool_calls > 0:
            web_search_calls = tool_calls
        if usage:
            yield TokenUsage(
                prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                web_search_calls=web_search_calls,
                tool_calls=tool_calls,
            )

    async def _stream_chat_completions(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str | TokenUsage, None]:
        """Fallback: Chat Completions API."""
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        usage = None
        async for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
            if chunk.usage:
                usage = chunk.usage
        if usage:
            yield TokenUsage(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            )

    def supported_models(self) -> list[str]:
        if self._live_models is not None:
            return list(self._live_models)
        return list(self.MODELS)
