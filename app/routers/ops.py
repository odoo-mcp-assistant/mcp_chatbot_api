"""Operational / infrastructure endpoints — health checks and config reload.

These routes are not part of the chat business logic and require no JWT auth.
They are called by monitoring tools and the Odoo addon, not by end users.
"""
import logging

from fastapi import APIRouter

from ..odoo_client import aodoo, get_client
from ..mcp_client import get_mcp
from ..odoo_config import get_odoo_config, load_odoo_config
from ..config import get_settings

_logger = logging.getLogger(__name__)

# no prefix — these routes are at the root level (e.g. /health, /reload_config)
router = APIRouter(tags=["ops"])


# simplest possible check — if this returns, the app process is alive
@router.get("/health")
async def health():
    return {"status": "ok"}


# verifies the Odoo connection is alive and authenticated
@router.get("/health/odoo")
async def health_odoo():
    odoo = get_client()                                      # get the connected Odoo client singleton
    user_name = await aodoo(lambda: odoo.env.user.name)      # ask Odoo for the logged-in user's name (in a thread)
    return {
        "status": "ok",
        "odoo_version": odoo.version,       # e.g. "16.0"
        "odoo_db": get_settings().odoo_db,  # the database name from .env
        "logged_in_as": user_name,          # confirms which Odoo user this app authenticated as
    }


# verifies the MCP server is connected and lists its available tools
@router.get("/health/mcp")
async def health_mcp():
    mcp = get_mcp()          # get the connected MCP client singleton
    cfg = get_odoo_config()  # get the cached chatbot config (contains the MCP server URL)
    return {
        "status": "ok",
        "server_url": cfg.mcp_server_url,                            # the URL of the MCP server
        "tool_count": len(mcp.tool_schemas),                         # how many tools the MCP server exposes
        "tools": [s["function"]["name"] for s in mcp.tool_schemas],  # list of tool names (e.g. get_orders)
    }


# shows the active chatbot configuration (no secrets exposed)
@router.get("/health/config")
async def health_config():
    cfg = get_odoo_config()  # get the cached chatbot config snapshot loaded from Odoo at startup
    return {
        "bot_name": cfg.bot_name,                      # the display name of the chatbot
        "status": cfg.status,                          # "active" or "inactive"
        "mcp_server_url": cfg.mcp_server_url,          # URL of the MCP server this app is connected to
        "llm_model": cfg.llm.model_name,               # the AI model being used
        "llm_base_url": cfg.llm.base_url,              # the API base URL for the LLM provider
        "summary_model": cfg.summary_llm.model_name,   # the model used to compress long conversation history
        "max_tool_rounds": cfg.max_tool_rounds,        # max times the AI can call tools in one message turn
        "summary_interval": cfg.summary_interval,      # how many tokens before history gets compressed
        "idle_timeout": cfg.idle_timeout,              # minutes of inactivity before a session is auto-closed
    }


# called by the Odoo addon after the admin saves chatbot settings
# re-reads all mcp_chatbot.* keys from Odoo and refreshes the in-memory config snapshot
@router.post("/reload_config")
async def reload_config():
    await aodoo(load_odoo_config)
    _logger.info("reload_config: config reloaded from Odoo")
    return {"status": "ok"}
