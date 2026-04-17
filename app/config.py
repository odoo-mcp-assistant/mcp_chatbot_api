"""
Runtime settings loaded from .env via pydantic-settings.

The .env file is the single source of truth for: Odoo connection,
FastAPI bind address, JWT secret, CORS, request timeout. Per-chatbot
settings (LLM API key, model, system prompt, MCP URL) still live in
Odoo's `ir.config_parameter` and are pulled via odoorpc at startup.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8020
    log_level: str = "info"

    # CORS — comma-separated list of allowed origins
    cors_origins: str = "http://localhost:8069"

    # Odoo connection (odoorpc → Odoo JSON-RPC)
    odoo_host: str = "localhost"
    odoo_port: int = 8069
    odoo_db: str
    odoo_user: str
    odoo_password: str

    # JWT
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_audience: str = "mcp-chatbot-api"

    # Agentic loop timeout (seconds)
    request_timeout: int = 118

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
