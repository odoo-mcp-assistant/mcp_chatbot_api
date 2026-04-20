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
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

_logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session: ClientSession | None = None
        self.tool_schemas: list[dict] = []
        self._exit_stack = AsyncExitStack()
        self._connected = False

    async def connect(self) -> None:
        if self._connected:
            return

        _logger.info("MCP: connecting to %s", self.server_url)

        # streamablehttp_client yields (read, write, get_session_id_callable)
        streams = await self._exit_stack.enter_async_context(
            streamable_http_client(self.server_url)
        )
        read, write, _ = streams

        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()

        tools = (await self.session.list_tools()).tools
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
        self._connected = True

        _logger.info(
            "MCP: connected, %d tools: %s",
            len(self.tool_schemas),
            [s["function"]["name"] for s in self.tool_schemas],
        )

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if not self._connected or self.session is None:
            raise RuntimeError("MCP client is not connected")
        return await self.session.call_tool(name, arguments=arguments or {})

    async def close(self) -> None:
        if not self._connected:
            return
        await self._exit_stack.aclose()
        self._connected = False
        self.session = None
        _logger.info("MCP: disconnected")


# Module-level singleton.
_client: MCPClient | None = None


async def init_mcp(server_url: str) -> MCPClient:
    global _client
    _client = MCPClient(server_url)
    await _client.connect()
    return _client


def get_mcp() -> MCPClient:
    if _client is None:
        raise RuntimeError("MCP client not initialised")
    return _client


async def close_mcp() -> None:
    global _client
    if _client is None:
        return
    await _client.close()
    _client = None
