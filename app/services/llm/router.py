from __future__ import annotations

from app.config import settings
from app.services.llm.anthropic import AnthropicProvider
from app.services.llm.base import LLMProvider
from app.services.llm.gemini import GeminiProvider
from app.services.llm.grok import GrokProvider
from app.services.llm.mistral import MistralProvider
from app.services.llm.openai import OpenAIProvider
from app.services.llm.status import provider_status_tracker

_PROVIDER_REGISTRY: list[tuple[str, type[LLMProvider], str]] = [
    ("openai_api_key", OpenAIProvider, "openai"),
    ("anthropic_api_key", AnthropicProvider, "anthropic"),
    ("google_ai_api_key", GeminiProvider, "google"),
    ("xai_api_key", GrokProvider, "xai"),
    ("mistral_api_key", MistralProvider, "mistral"),
]

_model_map: dict[str, LLMProvider] | None = None
_model_map_version: int = -1


def _init_providers() -> dict[str, LLMProvider]:
    """Create provider instances, register them with the status tracker,
    and build the initial model map from hardcoded fallback lists."""
    mapping: dict[str, LLMProvider] = {}
    for key_attr, cls, _ in _PROVIDER_REGISTRY:
        api_key: str = getattr(settings, key_attr)
        if not api_key:
            continue
        try:
            provider = cls(api_key=api_key)
        except ValueError:
            continue
        provider_status_tracker.register(provider)
        for model in provider.supported_models():
            mapping[model] = provider
    return mapping


def _rebuild_model_map() -> dict[str, LLMProvider]:
    """Rebuild the model map from already-registered providers using their
    current (possibly live-fetched) model lists."""
    mapping: dict[str, LLMProvider] = {}
    for provider in provider_status_tracker._providers.values():
        for model in provider.supported_models():
            mapping[model] = provider
    return mapping


def _get_model_map() -> dict[str, LLMProvider]:
    global _model_map, _model_map_version  # noqa: PLW0603
    if _model_map is None:
        _model_map = _init_providers()
        _model_map_version = provider_status_tracker.models_version
    elif _model_map_version != provider_status_tracker.models_version:
        _model_map = _rebuild_model_map()
        _model_map_version = provider_status_tracker.models_version
    return _model_map


def get_provider(model_name: str) -> LLMProvider:
    model_map = _get_model_map()
    provider = model_map.get(model_name)
    if provider is None:
        available = ", ".join(sorted(model_map.keys())) or "(none configured)"
        raise ValueError(
            f"No provider found for model '{model_name}'. "
            f"Available models: {available}"
        )
    return provider


def list_available_models() -> list[dict]:
    _get_model_map()
    result: list[dict] = []
    for provider in provider_status_tracker._providers.values():
        name = provider.provider_name()
        for model in provider.supported_models():
            result.append({"model": model, "provider": name})
    return result
