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
# init_mcp: connects to the MCP server at startup and fetches available tools
from .mcp_client import close_mcp, init_mcp

# connect: logs into Odoo at startup using credentials from .env
# disconnect: drops the Odoo connection at shutdown
from .odoo_client import connect, disconnect

from .odoo_config import load_odoo_config

from .routers import chat as chat_router
from .routers import ops as ops_router


# Configure the logging system: show INFO level and above, with timestamp + level + module name
logging.basicConfig(
    level=logging.INFO,                                     # show INFO, WARNING, ERROR (not DEBUG)
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

# Register all chat routes: POST /mcp_chatbot/message, POST /close, GET /history, GET /info
app.include_router(chat_router.router)

# Register all ops routes: GET /health, /health/odoo, /health/mcp, /health/config, POST /reload_config
app.include_router(ops_router.router)
