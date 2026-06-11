"""
Per-caller request throttling (Tier 0 abuse protection).

The chat widget is public: anyone — including a script that never logs in —
can obtain a JWT and POST to /mcp_chatbot/message, and every message costs
real LLM tokens. This module builds the shared ``slowapi`` limiter that caps
how fast a single caller may send messages, so a flood is rejected with HTTP
429 *before* it reaches the agent loop and spends anything.

Two pieces live here:

* ``rate_limit_key`` — decides what "a single caller" means for throttling.
* ``limiter`` — the shared Limiter instance. ``main.py`` registers it on the
  app and wires the 429 handler; ``routers/chat.py`` applies it to /message.

Why the key is identity-first, IP-second
-----------------------------------------
Anonymous visitors are identified by a ``session_token`` that the browser
generates itself (a fresh UUID), so an abuser can mint unlimited "new"
anonymous identities for free — throttling them by token would be pointless.
Their real, scarce identifier is the source IP, so anonymous callers are
keyed by IP. Logged-in users, by contrast, hold a partner_id that Odoo
vouched for, so they are keyed by partner_id — this keeps several genuine
users behind one shared office IP from throttling each other.
"""

import logging

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .auth import Principal, verify_token

_logger = logging.getLogger(__name__)


def _client_ip(request: Request) -> str:
    """Best-effort source IP, honouring a reverse proxy's X-Forwarded-For.

    When the service runs behind nginx / Cloudflare, ``request.client.host`` is
    the proxy, not the visitor, which would lump every anonymous caller into a
    single bucket. The left-most X-Forwarded-For entry is the original client.

    Caveat: X-Forwarded-For is client-spoofable unless a trusted proxy sets it.
    That is acceptable here because edge filtering (Tier 1) is the layer meant
    to sanitise this header; this throttle is a cheap second line of defence.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


def identity_key_for(principal: Principal, request: Request) -> str:
    """Canonical per-caller key shared by the rate limiter and the token
    budget: "partner:<id>" for logged-in users, "ip:<address>" otherwise.

    Keeping both features on the same key means a caller is throttled and
    budgeted as one identity. See the module docstring for why anonymous
    callers are keyed by IP rather than their (rotatable) session_token.
    """
    if principal.partner_id:
        return f"partner:{principal.partner_id}"
    return f"ip:{_client_ip(request)}"


def rate_limit_key(request: Request) -> str:
    """Throttling bucket for the current request (see module docstring).

    slowapi calls this with only the raw Request (no decoded Principal), so we
    verify the token here and delegate to identity_key_for for the actual key.
    """
    principal = Principal(partner_id=None, session_token=None)
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        try:
            principal = verify_token(auth[7:].strip())
        except HTTPException:
            # Invalid/expired token — fall through to IP. The endpoint's own
            # auth dependency will reject it with 401 anyway.
            pass
    return identity_key_for(principal, request)


# Shared limiter. key_func is the default bucket for every decorated route.
# Registered on the app in main.py (app.state.limiter + exception handler).
limiter = Limiter(key_func=rate_limit_key)


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """429 handler for a tripped rate limit.

    Replaces slowapi's default handler so the body carries a structured
    `detail.code` ("rate_limited") matching the budget 429's shape — letting
    the widget show "slow down" rather than the budget's "done for today".
    A Retry-After header is included so well-behaved clients can back off.
    """
    response = JSONResponse(
        status_code=429,
        content={
            "detail": {
                "code": "rate_limited",
                "message": "You're sending messages too quickly. "
                           "Please wait a few seconds and try again.",
            }
        },
    )
    # slowapi stamps request.state.view_rate_limit; mirror its Retry-After if set.
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        response.headers["Retry-After"] = str(retry_after)
    return response
