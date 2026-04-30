"""
Async MCP client.

Uses the MCP SDK's streamable HTTP transport natively — no thread bridge,
no `run_coroutine_threadsafe` dance like the original Odoo addon needed.
FastAPI already runs an asyncio event loop, so we just await things.

Lifecycle is owned by the FastAPI lifespan hook in main.py: connect at
startup, close at shutdown. The MCP session is a long-lived singleton
per worker process — all concurrent chat requests share it.
"""

import logging
# AsyncExitStack lets us register multiple async context managers (HTTP transport,
# ClientSession) and tear them all down in reverse order with a single aclose() call.
from contextlib import AsyncExitStack
# `Any` is used as the return type for tool calls since MCP tool results are dynamic.
from typing import Any

# ClientSession is the high-level MCP session object that wraps the JSON-RPC protocol over a pair of read/write streams.
from mcp import ClientSession
# streamable_http_client opens an HTTP-based bidirectional stream to the MCP server and yields the (read, write, get_session_id) triple expected by ClientSession.
from mcp.client.streamable_http import streamable_http_client

# Module-scoped logger; uses this module's dotted name so log filters can target it.
_logger = logging.getLogger(__name__)


class MCPClient:
    # Constructor — stores config and initializes empty state. No I/O happens here;
    # the actual network connection is deferred to `connect()` so the object can be safely created at import time and wired up later in the FastAPI lifespan hook.
    def __init__(self, server_url: str):
        self.server_url = server_url
        # The active MCP session, or None until connect() has run successfully.
        self.session: ClientSession | None = None
        # Cached OpenAI-style "function" tool schemas, populated after connect().
        # Kept on the client so callers can pass them straight to the LLM without
        # round-tripping to the MCP server on every chat request.
        self.tool_schemas: list[dict] = []
        # Stack of async context managers — owns both the HTTP streams and the session
        # so we can unwind them cleanly on shutdown via a single aclose().
        self._exit_stack = AsyncExitStack()
        # Idempotency flag so connect()/close() can be called more than once safely.
        self._connected = False

    # Opens the HTTP transport, initializes the MCP session, and caches tool schemas.
    async def connect(self) -> None:
        # Guard: if we're already connected, skip the whole setup. This protects against accidental double-initialization (e.g. lifespan re-entry, tests).
        if self._connected:
            return

        # Log the target URL so connection issues are easy to diagnose from logs.
        _logger.info("MCP: connecting to %s", self.server_url)

        # streamablehttp_client yields (read, write, get_session_id_callable)
        # Enter the streamable HTTP context manager *through the exit stack* so that
        # closing the stack later automatically tears the transport down.
        streams = await self._exit_stack.enter_async_context(
            streamable_http_client(self.server_url)
        )
        # Unpack the triple: `read` and `write` are the async stream halves used by
        # ClientSession; the third value (a session-id getter) isn't needed here.
        read, write, _ = streams

        # Create the MCP ClientSession on top of the streams and register it with the exit stack so it is also closed cleanly during shutdown.
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        # Perform the MCP `initialize` handshake — exchanges protocol versions and
        # capabilities with the server. Must happen before any tool calls.
        await self.session.initialize()

        # Ask the server which tools it exposes. `list_tools()` returns a response
        # object whose `.tools` attribute is the actual list of Tool descriptors.
        tools = (await self.session.list_tools()).tools
        # Re-shape MCP tool descriptors into the OpenAI/Chat Completions "function"
        # tool format so they can be passed verbatim to the LLM call. Each tool
        # contributes its name, human-readable description, and JSON-Schema params.
        self.tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            }
            for t in tools
        ]
        # Mark the client as ready — guards both the duplicate-connect path above
        # and the "not connected" check in call_tool().
        self._connected = True

        # Emit a summary log: how many tools we discovered and their names. Useful
        # for confirming the server exposes what we expect at startup.
        _logger.info(
            "MCP: connected, %d tools: %s",
            len(self.tool_schemas),
            [s["function"]["name"] for s in self.tool_schemas],
        )

    # Invoke a single MCP tool by name. `arguments` is the JSON-serializable payload
    # the LLM produced for this tool call; defaults to an empty dict for arg-less tools.
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        # Defensive check — calling a tool without an active session would otherwise
        # raise a less obvious AttributeError deep inside the SDK.
        if not self._connected or self.session is None:
            raise RuntimeError("MCP client is not connected")
        # Delegate to the SDK; `arguments or {}` normalizes None to an empty dict so
        # tools that take no parameters still get a valid JSON object on the wire.
        return await self.session.call_tool(name, arguments=arguments or {})

    # Tears down the session and HTTP transport. Idempotent — safe to call even if
    # connect() never ran or close() has already been called.
    async def close(self) -> None:
        # Nothing to do if we never connected (or already closed).
        if not self._connected:
            return
        # Closing the exit stack unwinds every context manager we registered,
        # in LIFO order: ClientSession first, then the HTTP transport.
        await self._exit_stack.aclose()
        # Reset internal state so a subsequent connect() can succeed cleanly.
        self._connected = False
        self.session = None
        # Final breadcrumb in the logs so a clean shutdown is visible.
        _logger.info("MCP: disconnected")


# Module-level singleton.
# Holds the single shared MCPClient for this worker process. Populated by
# init_mcp() during FastAPI startup; read by get_mcp() from request handlers.
_client: MCPClient | None = None


# Startup helper — constructs the singleton, connects it, and returns it. Called
# from main.py's lifespan handler so the connection is established before any
# request is served.
async def init_mcp(server_url: str) -> MCPClient:
    # `global` because we're rebinding the module-level name, not just mutating it.
    global _client
    # Build a fresh client. Replaces any previous instance — in normal operation
    # init_mcp() runs exactly once per process.
    _client = MCPClient(server_url)
    # Open the network connection and populate tool schemas.
    await _client.connect()
    # Return the client to the caller as a convenience (the singleton is also
    # available via get_mcp()).
    return _client


# Accessor used by request handlers to reach the singleton. Raises if the lifespan
# hook hasn't run yet — better to fail loudly than to lazy-connect mid-request.
def get_mcp() -> MCPClient:
    if _client is None:
        raise RuntimeError("MCP client not initialised")
    return _client


# Shutdown helper — mirrors init_mcp(). Called from the lifespan hook on app exit
# to release the HTTP connection and inner session cleanly.
async def close_mcp() -> None:
    global _client
    # No-op if init_mcp() never ran (e.g. startup failed early).
    if _client is None:
        return
    # Delegate the actual teardown to the client.
    await _client.close()
    # Drop the reference so any later get_mcp() call fails fast instead of returning
    # a half-closed object.
    _client = None


# ---------------------------------------------------------------------------
# Background notes — concepts used in this module
# ---------------------------------------------------------------------------
#
# 1. Context managers
# -------------------
# A context manager is any object you can use with `with` (sync) or `async with`
# (async). It defines two phases:
#   - setup    → __enter__  / __aenter__   (runs when entering the block)
#   - teardown → __exit__   / __aexit__    (runs when leaving the block, even
#                                           if an exception was raised)
#
# It's just a structured way to say "do X now, and guarantee Y happens
# afterward". Example:
#
#     with open("file.txt") as f:        # __enter__: opens the file
#         f.read()
#     # __exit__: closes the file, automatically
#
# The async version uses `async with` and exists for resources that need to
# await during setup/teardown (network sockets, DB pools, MCP sessions, …).
#
#
# 2. AsyncExitStack
# -----------------
# `async with` only keeps a resource alive for the duration of one block.
# That's a problem here: we want the MCP transport + session to be opened in
# `connect()` and closed later in `close()` — two different methods.
#
# `AsyncExitStack` solves this. It lets us *manually* enter context managers
# now and defer their teardown until we explicitly ask for it:
#
#     stack = AsyncExitStack()
#     value = await stack.enter_async_context(some_cm())   # __aenter__ runs now
#     # ... time passes, other code runs ...
#     await stack.aclose()                                  # __aexit__ runs now
#
# It's a *stack*: you can register multiple context managers, and `aclose()`
# tears them down in LIFO order — the same order nested `async with` blocks
# would unwind. Here we push the HTTP transport first and the ClientSession
# second, so on shutdown the session closes before the transport (which is
# the correct order — closing the transport first would yank the streams out
# from under the session).
#
#
# 3. The `streams` triple
# -----------------------
# `streamable_http_client(url)` is an async context manager that, on entry,
# opens an HTTP connection to the MCP server and yields a 3-tuple:
#
#     (read, write, get_session_id)
#
#   - read              : async stream of incoming JSON-RPC messages from the
#                         server (tool results, responses, notifications).
#   - write             : async stream for outgoing JSON-RPC messages (the
#                         initialize handshake, tools/list, tools/call, …).
#   - get_session_id    : a callable returning the server-assigned session id
#                         (a string), or None before the server has issued
#                         one. Useful for logging/correlation; we don't need
#                         it here, so we discard it with `_`.
#
# `ClientSession(read, write)` wraps those two streams and speaks the MCP
# wire protocol on top of them, so callers only deal with high-level methods
# like `initialize()`, `list_tools()`, and `call_tool()`.
#
#
# 4. How the MCP Python SDK uses these patterns
# ---------------------------------------------
# Both halves of the SDK lean heavily on context managers:
#
#   Client side — every transport (`streamable_http_client`, `stdio_client`,
#   `sse_client`) and `ClientSession` itself are async context managers. The
#   SDK's documented usage is two nested `async with` blocks:
#
#       async with streamable_http_client(url) as (read, write, _):
#           async with ClientSession(read, write) as session:
#               await session.initialize()
#               ...
#
#   This module uses `AsyncExitStack` to flatten that nesting so the lifetime
#   spans `connect()`/`close()` instead of one function.
#
#   Server side — `FastMCP` uses `@asynccontextmanager` for its `lifespan`
#   hook (yield-based setup/teardown for shared resources like DB pools), and
#   `mcp.run()` internally enters its transport's context manager.
#
# The reason the SDK is built this way: MCP sessions involve background async
# tasks (reader, writer, dispatcher). Context managers force structured,
# exception-safe cleanup, which prevents leaked tasks and hung connections.
# ---------------------------------------------------------------------------
