"""
mcp_chatbot_api — FastAPI sidecar for the Odoo chatbot.

The browser widget calls this service directly for chat, history, close,
and info. Odoo is only involved for settings UI and JWT issuance; all
chat-request DB I/O goes through odoorpc (wrapped in asyncio.to_thread)
so Odoo worker load per message drops from ~10-30 s to ~250-500 ms.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .mcp_client import close_mcp, get_mcp, init_mcp
from .odoo_client import aodoo, connect, disconnect, get_client
from .odoo_config import get_odoo_config, load_odoo_config
from .routers import chat as chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup — fail loud on any wiring issue.
    connect()                         # odoorpc login
    cfg = load_odoo_config()          # snapshot mcp_chatbot.* from Odoo
    await init_mcp(cfg.mcp_server_url)  # connect to MCP server
    _logger.info("startup: ready")
    yield
    # Shutdown
    await close_mcp()
    disconnect()


app = FastAPI(
    title="MCP Chatbot API",
    version="0.1.0",
    description=(
        "FastAPI sidecar for the Odoo MCP chatbot. "
        "Handles the async LLM + MCP agentic loop outside the Odoo "
        "worker pool so chat requests do not compete with checkout traffic."
    ),
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/odoo")
async def health_odoo():
    """Verify the odoorpc client is connected and Odoo responds."""
    odoo = get_client()
    user_name = await aodoo(lambda: odoo.env.user.name)
    return {
        "status": "ok",
        "odoo_version": odoo.version,
        "odoo_db": get_settings().odoo_db,
        "logged_in_as": user_name,
    }


@app.get("/health/mcp")
async def health_mcp():
    """Verify the MCP client is connected and advertises tools."""
    mcp = get_mcp()
    cfg = get_odoo_config()
    return {
        "status": "ok",
        "server_url": cfg.mcp_server_url,
        "tool_count": len(mcp.tool_schemas),
        "tools": [s["function"]["name"] for s in mcp.tool_schemas],
    }


@app.get("/health/config")
async def health_config():
    """Return the active chatbot config snapshot (no secrets)."""
    cfg = get_odoo_config()
    return {
        "bot_name": cfg.bot_name,
        "status": cfg.status,
        "mcp_server_url": cfg.mcp_server_url,
        "llm_model": cfg.llm.model_name,
        "llm_base_url": cfg.llm.base_url,
        "summary_model": cfg.summary_llm.model_name,
        "max_tool_rounds": cfg.max_tool_rounds,
        "summary_interval": cfg.summary_interval,
        "idle_timeout": cfg.idle_timeout,
    }
