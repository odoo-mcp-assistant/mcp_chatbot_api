"""
Runtime settings loaded from .env via pydantic-settings.

The .env file is the single source of truth for: Odoo connection,
FastAPI bind address, JWT secret, CORS, request timeout. Per-chatbot
settings (LLM API key, model, system prompt, MCP URL) still live in
Odoo's `ir.config_parameter` and are pulled via odoorpc at startup.
"""

# lru_cache: caches the result of a function in memory so it's only computed once
# used here so the .env file is only read once at startup, not on every request
from functools import lru_cache

# BaseSettings: pydantic class that reads values from environment variables and .env files
# SettingsConfigDict: used to configure how BaseSettings behaves (which file to read, encoding, etc.)
from pydantic_settings import BaseSettings, SettingsConfigDict


# Settings inherits from BaseSettings — pydantic automatically reads each field
# from the .env file or from actual environment variables
class Settings(BaseSettings):

    # model_config tells pydantic-settings HOW to load the settings
    model_config = SettingsConfigDict(
        env_file=".env",            # read values from a file named .env in the project root
        env_file_encoding="utf-8",  # the .env file is encoded in UTF-8
        case_sensitive=False,       # ODOO_HOST and odoo_host are treated as the same variable
        extra="ignore",             # if .env has extra variables we don't define here, silently ignore them
    )

    # --- FastAPI server settings ---
    api_host: str = "0.0.0.0"   # the network interface to bind to — 0.0.0.0 means accept connections from anywhere
    api_port: int = 8020         # the port the FastAPI server listens on
    log_level: str = "info"      # the logging verbosity level passed to uvicorn

    # --- CORS setting ---
    # comma-separated list of allowed origins, e.g. "https://myshop.odoo.com,http://localhost:8069"
    # default is localhost Odoo for local development
    cors_origins: str = "http://localhost:8069"

    # --- Odoo connection settings ---
    # these are used by odoorpc to log into Odoo at startup
    odoo_host: str = "localhost"  # hostname or IP of the Odoo server
    odoo_port: int = 8069         # port Odoo is running on (default Odoo port)
    odoo_db: str                  # the Odoo database name — NO default, must be set in .env
    odoo_user: str                # the Odoo username this app logs in as — NO default, must be set in .env
    odoo_password: str            # the Odoo password — NO default, must be set in .env

    # --- JWT settings ---
    # the JWT secret must match the value stored in Odoo's ir.config_parameter['mcp_chatbot.jwt_secret']
    # because Odoo mints the token and this app verifies it — they must share the same secret
    jwt_secret: str               # the shared secret used to sign and verify JWT tokens — NO default, must be set in .env
    jwt_algorithm: str = "HS256"  # the signing algorithm — HS256 is the standard symmetric algorithm
    jwt_audience: str = "mcp-chatbot-api"  # the expected "aud" claim in the token — must match what Odoo puts in the token

    # --- Agentic loop timeout ---
    # 118 seconds = just under 2 minutes, intentionally just under nginx/proxy default timeout of 120s
    # if the AI loop takes longer than this, the request is cancelled
    request_timeout: int = 118

    # cors_origins_list is a computed property — it converts the raw comma-separated string
    # into a Python list that CORSMiddleware can actually use
    # e.g. "https://a.com, https://b.com" → ["https://a.com", "https://b.com"]
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# @lru_cache makes this function only run ONCE — after the first call, the result
# is stored in memory and returned directly on every subsequent call
# this means the .env file is parsed exactly once at startup, not on every request
@lru_cache
def get_settings() -> Settings:
    return Settings()  # pydantic reads the .env file and populates all fields above
