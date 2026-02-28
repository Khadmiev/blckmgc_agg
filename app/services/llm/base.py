from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class LLMProvider(ABC):
    @abstractmethod
    async def stream_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Yield text chunks as they arrive from the LLM."""
        ...

    @abstractmethod
    def supported_models(self) -> list[str]:
        """Return list of model identifiers this provider supports."""
        ...

    @abstractmethod
    async def fetch_models(self) -> list[str]:
        """Fetch available models from the provider API and update internal state.

        Returns the current model list (live if fetch succeeds, fallback otherwise).
        Must never raise — failures are handled internally.
        """
        ...

    @abstractmethod
    async def health_check(self) -> None:
        """Verify connectivity to the provider. Raise on failure."""
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Return a stable identifier for this provider (e.g. 'openai')."""
        ...
