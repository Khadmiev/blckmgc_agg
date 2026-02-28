from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMProvider(ABC):
    @abstractmethod
    async def stream_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str | TokenUsage, None]:
        """Yield text chunks as they arrive, followed by a final TokenUsage."""
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
