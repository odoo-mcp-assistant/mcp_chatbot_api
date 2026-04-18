"""
mcp_chatbot_api — FastAPI sidecar for the Odoo chatbot.

The browser widget calls this service directly for chat, history, close,
and info. Odoo is only involved for settings UI and JWT issuance; all
chat-request DB I/O goes through odoorpc (wrapped in asyncio.to_thread)
so Odoo worker load per message drops from ~10-30 s to ~250-500 ms.
"""

# logging: Python's built-in module for printing info/warning/error messages
import logging

# asynccontextmanager: lets us write the startup/shutdown logic as a single function with yield
from contextlib import asynccontextmanager

# FastAPI: the main class that represents our web application
from fastapi import FastAPI

# CORSMiddleware: the piece that allows the Odoo browser widget to call this API
from fastapi.middleware.cors import CORSMiddleware

# get_settings: reads our .env file and returns all infrastructure config (ports, passwords, etc.)
from .config import get_settings

# close_mcp: disconnects from the MCP server at shutdown
# get_mcp: returns the already-connected MCP client (used in health check)
# init_mcp: connects to the MCP server at startup and fetches available tools
from .mcp_client import close_mcp, get_mcp, init_mcp

# aodoo: runs a synchronous Odoo call in a background thread so it doesn't block the app
# connect: logs into Odoo at startup using credentials from .env
# disconnect: drops the Odoo connection at shutdown
# get_client: returns the already-connected Odoo client (used in health check)
from .odoo_client import aodoo, connect, disconnect, get_client

# get_odoo_config: returns the already-loaded chatbot config snapshot from Odoo
# load_odoo_config: reads chatbot settings (LLM model, system prompt, MCP URL, etc.) from Odoo at startup
from .odoo_config import get_odoo_config, load_odoo_config

# chat_router: the file that contains all /mcp_chatbot/* endpoints (message, history, close, info)
from .routers import chat as chat_router


# Configure the logging system: show INFO level and above, with timestamp + level + module name
logging.basicConfig(
    level=logging.INFO,                                    # show INFO, WARNING, ERROR (not DEBUG)
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # format: "2024-01-01 [INFO] app.main: ..."
)

# Create a logger for this specific file — messages will show "app.main" as the source
_logger = logging.getLogger(__name__)


# @asynccontextmanager turns this function into a startup/shutdown manager
# Everything before `yield` runs at startup, everything after runs at shutdown
@asynccontextmanager
async def lifespan(_: FastAPI):   # receives the app instance but we don't need it (hence _)

    # STARTUP — these 3 lines run once when the app boots
    connect()                          # log into Odoo using credentials from .env — crashes loud if it fails
    cfg = load_odoo_config()           # read chatbot settings from Odoo's ir.config_parameter table
    await init_mcp(cfg.mcp_server_url) # open a persistent connection to the MCP server and fetch its tools

    _logger.info("startup: ready")     # print "startup: ready" to the logs so we know the app is live

    yield                              # ← the app is now running and serving requests — pause here until shutdown

    # SHUTDOWN — these 2 lines run once when the app is stopped (Ctrl+C or process kill)
    await close_mcp()   # gracefully close the MCP server connection
    disconnect()        # drop the Odoo client reference


# Create the FastAPI application object — this is the core of the whole API
app = FastAPI(
    title="MCP Chatbot API",       # shown in the auto-generated API docs at /docs
    description=(                  # shown in the auto-generated API docs
        "FastAPI sidecar for the Odoo MCP chatbot. "
        "Handles the async LLM + MCP agentic loop outside the Odoo "
        "worker pool so chat requests do not compete with checkout traffic."
    ),
    lifespan=lifespan,             # wire up the startup/shutdown function we defined above
)

# Read the .env settings once and store them — we need cors_origins_list below
_settings = get_settings()

# Register the CORS middleware — this must be added before any routes
app.add_middleware(
    CORSMiddleware,                              # the middleware class that handles CORS
    allow_origins=_settings.cors_origins_list,   # only allow requests from these domains (e.g. your Odoo URL)
    allow_credentials=True,                      # allow the Authorization: Bearer header (needed for JWT)
    allow_methods=["*"],                         # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],                         # allow all headers (including Authorization)
)

# Register all routes defined in routers/chat.py under the /mcp_chatbot prefix
# This adds: POST /mcp_chatbot/message, POST /mcp_chatbot/history, etc.
app.include_router(chat_router.router)



























## All the endpoints below are for testing health of other services and are not included in business logic 







# Health check #1 — the simplest possible check: is the app process alive?
@app.get("/health")
async def health():
    return {"status": "ok"}   # if this returns, the app is running


# Health check #2 — verifies the Odoo connection is alive and authenticated
@app.get("/health/odoo")
async def health_odoo():
    odoo = get_client()                            # get the connected Odoo client singleton
    user_name = await aodoo(lambda: odoo.env.user.name)  # ask Odoo for the logged-in user's name (in a thread)
    return {
        "status": "ok",
        "odoo_version": odoo.version,              # e.g. "16.0"
        "odoo_db": get_settings().odoo_db,         # the database name from .env
        "logged_in_as": user_name,                 # confirms which Odoo user this app authenticated as
    }


# Health check #3 — verifies the MCP server is connected and lists its available tools
@app.get("/health/mcp")
async def health_mcp():
    mcp = get_mcp()          # get the connected MCP client singleton
    cfg = get_odoo_config()  # get the cached chatbot config (contains the MCP server URL)
    return {
        "status": "ok",
        "server_url": cfg.mcp_server_url,                              # the URL of the MCP server
        "tool_count": len(mcp.tool_schemas),                           # how many tools the MCP server exposes
        "tools": [s["function"]["name"] for s in mcp.tool_schemas],   # list of tool names (e.g. get_orders)
    }


# Health check #4 — shows the active chatbot configuration (no secrets exposed)
@app.get("/health/config")
async def health_config():
    cfg = get_odoo_config()   # get the cached chatbot config snapshot loaded from Odoo at startup
    return {
        "bot_name": cfg.bot_name,              # the display name of the chatbot (e.g. "Aria")
        "status": cfg.status,                  # "active" or "inactive" — controls whether the widget responds
        "mcp_server_url": cfg.mcp_server_url,  # URL of the MCP server this app is connected to
        "llm_model": cfg.llm.model_name,       # the AI model being used (e.g. "gpt-4o")
        "llm_base_url": cfg.llm.base_url,      # the API base URL for the LLM provider
        "summary_model": cfg.summary_llm.model_name,   # the model used to compress long conversation history
        "max_tool_rounds": cfg.max_tool_rounds,        # max times the AI can call tools in one message turn
        "summary_interval": cfg.summary_interval,      # how many tokens before history gets compressed
        "idle_timeout": cfg.idle_timeout,              # minutes of inactivity before a session is auto-closed
    }
