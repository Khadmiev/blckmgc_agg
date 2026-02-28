from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)

STALE_THRESHOLD_SECONDS = 3600  # 1 hour
BACKGROUND_CHECK_INTERVAL_SECONDS = 300  # poll every 5 minutes
MODEL_REFRESH_INTERVAL_SECONDS = 3600  # refresh model lists every hour


@dataclass
class ProviderStatus:
    provider_name: str
    available: bool = False
    models: list[str] = field(default_factory=list)
    last_checked: datetime | None = None
    last_success: datetime | None = None
    last_model_refresh: datetime | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider_name,
            "available": self.available,
            "models": self.models,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_model_refresh": (
                self.last_model_refresh.isoformat() if self.last_model_refresh else None
            ),
            "error": self.error,
        }


class ProviderStatusTracker:
    def __init__(self) -> None:
        self._statuses: dict[str, ProviderStatus] = {}
        self._providers: dict[str, LLMProvider] = {}
        self._background_task: asyncio.Task | None = None
        self._models_version: int = 0

    @property
    def models_version(self) -> int:
        return self._models_version

    def register(self, provider: LLMProvider) -> None:
        name = provider.provider_name()
        self._providers[name] = provider
        self._statuses[name] = ProviderStatus(
            provider_name=name,
            models=provider.supported_models(),
        )

    async def check_provider(self, name: str) -> bool:
        provider = self._providers.get(name)
        status = self._statuses.get(name)
        if provider is None or status is None:
            return False

        now = datetime.now(timezone.utc)
        status.last_checked = now
        try:
            await provider.health_check()
            status.available = True
            status.last_success = now
            status.error = None
            logger.info("Provider %s: healthy", name)
            return True
        except Exception as exc:
            status.available = False
            status.error = str(exc)
            logger.warning("Provider %s: health check failed — %s", name, exc)
            return False

    async def check_all(self) -> None:
        """Run health checks for all registered providers concurrently."""
        tasks = [self.check_provider(name) for name in self._providers]
        await asyncio.gather(*tasks, return_exceptions=True)

    def record_success(self, provider_name: str) -> None:
        """Called after a successful LLM completion to extend availability."""
        status = self._statuses.get(provider_name)
        if status is None:
            return
        now = datetime.now(timezone.utc)
        status.available = True
        status.last_success = now
        status.last_checked = now
        status.error = None

    def record_failure(self, provider_name: str, error: str) -> None:
        """Called after a failed LLM completion."""
        status = self._statuses.get(provider_name)
        if status is None:
            return
        status.last_checked = datetime.now(timezone.utc)
        status.error = error

    async def refresh_models_for(self, name: str) -> None:
        """Fetch the live model list for a single provider."""
        provider = self._providers.get(name)
        status = self._statuses.get(name)
        if provider is None or status is None:
            return
        models = await provider.fetch_models()
        status.models = models
        status.last_model_refresh = datetime.now(timezone.utc)

    async def refresh_all_models(self) -> None:
        """Fetch live model lists from all providers concurrently."""
        tasks = [self.refresh_models_for(name) for name in self._providers]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._models_version += 1
        logger.info("Model lists refreshed (version %d)", self._models_version)

    async def _check_stale_providers(self) -> None:
        """Re-check providers that haven't had activity for STALE_THRESHOLD_SECONDS."""
        now = datetime.now(timezone.utc)
        for name, status in self._statuses.items():
            if status.last_success is None:
                continue
            elapsed = (now - status.last_success).total_seconds()
            if elapsed >= STALE_THRESHOLD_SECONDS:
                logger.info("Provider %s is stale (%.0fs since last success), re-checking", name, elapsed)
                await self.check_provider(name)

    def _model_refresh_due(self) -> bool:
        """Check if any provider hasn't had a model refresh within the interval."""
        now = datetime.now(timezone.utc)
        for status in self._statuses.values():
            if status.last_model_refresh is None:
                return True
            elapsed = (now - status.last_model_refresh).total_seconds()
            if elapsed >= MODEL_REFRESH_INTERVAL_SECONDS:
                return True
        return False

    async def _background_loop(self) -> None:
        """Periodically check stale providers and refresh model lists."""
        while True:
            await asyncio.sleep(BACKGROUND_CHECK_INTERVAL_SECONDS)
            try:
                await self._check_stale_providers()
            except Exception:
                logger.exception("Error in provider status background loop")
            try:
                if self._model_refresh_due():
                    await self.refresh_all_models()
            except Exception:
                logger.exception("Error in model refresh background loop")

    def start_background_checks(self) -> None:
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._background_loop())

    def stop_background_checks(self) -> None:
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()

    def get_all_statuses(self) -> list[dict]:
        return [s.to_dict() for s in self._statuses.values()]

    def get_status(self, provider_name: str) -> dict | None:
        status = self._statuses.get(provider_name)
        return status.to_dict() if status else None

    def is_available(self, provider_name: str) -> bool:
        status = self._statuses.get(provider_name)
        return status.available if status else False


provider_status_tracker = ProviderStatusTracker()
