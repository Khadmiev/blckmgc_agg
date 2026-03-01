from __future__ import annotations

import logging
from typing import AsyncGenerator

from mistralai import Mistral

from app.config import settings
from app.services.llm.base import LLMProvider, TokenUsage

logger = logging.getLogger(__name__)

_EXCLUDED_PREFIXES = ("mistral-embed",)
_WEB_SEARCH_TOOL = {"type": "web_search"}


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
            if settings.use_response_apis:
                async for item in self._stream_conversations(
                    messages, model, temperature, max_tokens
                ):
                    yield item
            else:
                async for item in self._stream_chat(
                    messages, model, temperature, max_tokens
                ):
                    yield item
        except Exception as exc:
            if "mistral" in type(exc).__module__.lower():
                raise Exception(f"Mistral API error: {exc}") from exc
            raise

    async def _stream_conversations(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str | TokenUsage, None]:
        """Use Beta Conversations API with web search when available."""
        beta = getattr(self.client, "beta", None)
        if beta is None:
            logger.debug("Mistral: beta API not available, using Chat")
            async for item in self._stream_chat(messages, model, temperature, max_tokens):
                yield item
            return
        conv = getattr(beta, "conversations", None)
        if conv is None:
            async for item in self._stream_chat(messages, model, temperature, max_tokens):
                yield item
            return
        start_stream = getattr(conv, "start_stream", None)
        if start_stream is None:
            async for item in self._stream_chat(messages, model, temperature, max_tokens):
                yield item
            return
        inputs = self._messages_to_conversation_inputs(messages)
        try:
            stream = start_stream(
                model=model,
                inputs=inputs,
                tools=[_WEB_SEARCH_TOOL],
                stream=True,
                completion_args={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
        except (AttributeError, TypeError, Exception) as e:
            logger.warning("Mistral Conversations API failed, falling back to Chat: %s", e)
            async for item in self._stream_chat(messages, model, temperature, max_tokens):
                yield item
            return
        full_text: list[str] = []
        usage = None
        async for event in stream:
            if hasattr(event, "data") and event.data:
                d = event.data
                if hasattr(d, "content") and d.content:
                    full_text.append(d.content)
                    yield d.content
                if hasattr(d, "usage") and d.usage:
                    usage = d.usage
            elif hasattr(event, "content") and event.content:
                full_text.append(event.content)
                yield event.content
            elif hasattr(event, "usage") and event.usage:
                usage = event.usage
        web_search_calls = 0
        tool_calls = 0
        if usage:
            connectors = getattr(usage, "connectors", None)
            if connectors and isinstance(connectors, dict):
                for _key, count in connectors.items():
                    n = count if isinstance(count, int) else 0
                    tool_calls += n
                    if "web_search" in str(_key).lower():
                        web_search_calls += n
            if web_search_calls == 0 and tool_calls > 0:
                web_search_calls = tool_calls
        if usage:
            yield TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                web_search_calls=web_search_calls,
                tool_calls=tool_calls,
            )

    def _messages_to_conversation_inputs(self, messages: list[dict]) -> str | list:
        """Convert chat messages to Conversations API inputs.
        Single user message: string. Multi-turn: array of {role, content}."""
        filtered = [m for m in messages if m.get("role") != "system"]
        if not filtered:
            return ""
        if len(filtered) == 1 and filtered[0].get("role") == "user":
            c = filtered[0].get("content", "")
            return c if isinstance(c, str) else str(c)
        entries = []
        for m in filtered:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                entries.append({"role": role, "content": content})
            else:
                entries.append({"role": role, "content": str(content)})
        return entries

    async def _stream_chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str | TokenUsage, None]:
        """Chat Completions API (no tools)."""
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

    def supported_models(self) -> list[str]:
        if self._live_models is not None:
            return list(self._live_models)
        return list(self.MODELS)
