"""
odoorpc client wrapper.

Design choices
--------------
- **One client per FastAPI worker process.** Authenticated once at
  startup. Reused for every request. odoorpc's internal `requests.Session`
  is used across threads — JSON-RPC calls are stateless with respect to
  cookies (the session cookie identifies the logged-in user, not a
  per-request transaction), so sharing is safe for typical read/write ops.
- **Sync library, async app.** odoorpc is built on `requests`, which is
  synchronous. Every call must be wrapped in `asyncio.to_thread()` so it
  runs in the default thread pool instead of blocking the event loop.
  The helper `aodoo()` below does that.
- **Fail loud at startup.** If the connection / login fails, the FastAPI
  app refuses to boot. Better to fail immediately than have every chat
  request error out at runtime.
"""

import asyncio
import logging
from typing import Any, Callable, TypeVar

import odoorpc

from .config import get_settings

_logger = logging.getLogger(__name__)

T = TypeVar("T")

# Module-level singleton — lives for the life of the worker process.
_odoo: odoorpc.ODOO | None = None


def connect() -> odoorpc.ODOO:
    """
    Create and authenticate the odoorpc client. Idempotent — calling
    twice returns the same client. Raises on failure.
    """
    global _odoo

    if _odoo is not None:
        return _odoo

    settings = get_settings()
    _logger.info(
        "odoorpc: connecting to %s:%s db=%s as %s",
        settings.odoo_host, settings.odoo_port, settings.odoo_db, settings.odoo_user,
    )

    client = odoorpc.ODOO(
        settings.odoo_host,
        port=settings.odoo_port,
    )
    client.login(settings.odoo_db, settings.odoo_user, settings.odoo_password)

    _logger.info(
        "odoorpc: logged in as uid=%s (%s), Odoo version=%s",
        client.env.uid,
        client.env.user.name,
        client.version,
    )

    _odoo = client
    return client


def get_client() -> odoorpc.ODOO:
    """Return the connected client. Raises if connect() was not called."""
    if _odoo is None:
        raise RuntimeError(
            "odoorpc client is not initialised — call connect() at startup"
        )
    return _odoo


async def aodoo(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Run a synchronous odoorpc call in a thread so it doesn't block the
    FastAPI event loop.

    Usage:
        Session = odoo.env['mcp.chatbot.session']
        session_id = await aodoo(Session.create, {'partner_id': 42})
        vals = await aodoo(Session.browse(session_id).read, ['name', 'state'])
    """
    return await asyncio.to_thread(fn, *args, **kwargs)


def disconnect() -> None:
    """Drop the client reference. odoorpc has no explicit logout."""
    global _odoo
    _odoo = None
    _logger.info("odoorpc: client disconnected")
