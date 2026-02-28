from __future__ import annotations

import json
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "LPO"
    debug: bool = False
    secret_key: str = "change-me-to-a-random-secret"
    api_v1_prefix: str = "/api/v1"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/lpo"

    # JWT
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    apple_client_id: str = ""
    apple_team_id: str = ""
    apple_key_id: str = ""
    apple_private_key: str = ""

    # LLM API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_ai_api_key: str = ""
    xai_api_key: str = ""
    mistral_api_key: str = ""

    # Storage
    storage_backend: str = "local"
    media_dir: str = "media"
    media_url_prefix: str = "/api/v1/media"

    # CORS
    cors_origins: str = '["*"]'

    @property
    def cors_origin_list(self) -> list[str]:
        return json.loads(self.cors_origins)

    @property
    def media_path(self) -> Path:
        return Path(self.media_dir)


settings = Settings()
