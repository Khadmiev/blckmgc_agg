from __future__ import annotations

import base64
import logging
from typing import AsyncGenerator

from google import genai
from google.genai import types

from app.services.llm.base import LLMProvider, TokenUsage

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro-preview-06-05"]

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Google AI API key is not configured")
        self._api_key = api_key
        self._client: genai.Client | None = None
        self._live_models: list[str] | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def provider_name(self) -> str:
        return "google"

    async def health_check(self) -> None:
        try:
            await self.client.aio.models.get(model=self.supported_models()[0])
        except Exception as exc:
            raise Exception(f"Google AI health check failed: {exc}") from exc

    async def fetch_models(self) -> list[str]:
        try:
            models: list[str] = []
            async for m in await self.client.aio.models.list():
                methods = getattr(m, "supported_generation_methods", []) or []
                if "generateContent" not in methods:
                    continue
                name = m.name or ""
                short_name = name.removeprefix("models/")
                if short_name.startswith("gemini-"):
                    models.append(short_name)
            if models:
                self._live_models = sorted(models)
                logger.info("Gemini: fetched %d models", len(models))
        except Exception:
            logger.warning("Gemini: failed to fetch models, using fallback", exc_info=True)
        return self._live_models if self._live_models is not None else list(self.MODELS)

    def _convert_messages(
        self, messages: list[dict]
    ) -> tuple[list[types.Content], str | None]:
        """Convert standard chat messages to Gemini contents format."""
        contents: list[types.Content] = []
        system_instruction: str | None = None

        for msg in messages:
            role = msg["role"]
            raw = msg["content"]

            if role == "system":
                system_instruction = raw if isinstance(raw, str) else str(raw)
                continue

            gemini_role = "model" if role == "assistant" else "user"

            if isinstance(raw, str):
                parts = [types.Part(text=raw)]
            else:
                parts = []
                for block in raw:
                    if block["type"] == "text":
                        parts.append(types.Part(text=block["text"]))
                    elif block["type"] == "image_url":
                        url = block["image_url"]["url"]
                        if url.startswith("data:"):
                            mime, _, b64 = url.partition(";base64,")
                            mime = mime.removeprefix("data:")
                            parts.append(
                                types.Part(
                                    inline_data=types.Blob(
                                        mime_type=mime,
                                        data=base64.b64decode(b64),
                                    )
                                )
                            )
                        else:
                            parts.append(types.Part(text=f"[Image: {url}]"))

            contents.append(types.Content(role=gemini_role, parts=parts))

        return contents, system_instruction

    async def stream_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str | TokenUsage, None]:
        try:
            contents, system_instruction = self._convert_messages(messages)

            config_kwargs: dict = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction

            config = types.GenerateContentConfig(**config_kwargs)

            usage_meta = None
            async for chunk in self.client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    yield chunk.text
                if getattr(chunk, "usage_metadata", None):
                    usage_meta = chunk.usage_metadata
            if usage_meta:
                prompt = getattr(usage_meta, "prompt_token_count", 0) or 0
                completion = getattr(usage_meta, "candidates_token_count", 0) or 0
                total = getattr(usage_meta, "total_token_count", 0) or (prompt + completion)
                yield TokenUsage(
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=total,
                )
        except Exception as exc:
            if "google" in type(exc).__module__:
                raise Exception(f"Google AI API error: {exc}") from exc
            raise

    def supported_models(self) -> list[str]:
        if self._live_models is not None:
            return list(self._live_models)
        return list(self.MODELS)
