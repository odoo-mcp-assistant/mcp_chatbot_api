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

import ipaddress
import logging

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .auth import Principal, verify_token
from .config import get_settings

_logger = logging.getLogger(__name__)


def _is_trusted_proxy(ip: str) -> bool:
    """True when `ip` belongs to a reverse proxy we control (TRUSTED_PROXIES).

    Anything unparseable is untrusted — failing closed here can only make us
    ignore a forwarding header, never believe a forged one.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in get_settings().trusted_proxy_networks)


def _client_ip(request: Request) -> str:
    """The visitor's real source IP, honouring forwarding headers only when
    they come from a proxy we trust.

    Forwarding headers (X-Forwarded-For, CF-Connecting-IP) are plain text the
    client can type, so believing them unconditionally would let an abuser
    mint a fresh fake "IP" per request — bypassing both the per-IP rate limit
    and the anonymous daily budget. The trust rule:

    1. If the TCP peer is NOT in TRUSTED_PROXIES, the request reached us
       directly: use the socket address, ignore every forwarding header.
    2. If the peer IS trusted and CLIENT_IP_HEADER is configured (behind
       Cloudflare: CF-Connecting-IP, which the edge overwrites on every
       request), use that header.
    3. Otherwise walk X-Forwarded-For right-to-left and return the first hop
       that isn't a trusted proxy — hops to the left of that are claims
       written by the client, not observations made by our infrastructure.

    Note the edge must actually be exclusive for (2) to hold: if the origin
    accepts traffic from anywhere (no firewall allow-list), an attacker can
    bypass Cloudflare and forge its header from a "trusted" local path. See
    docs/cloudflare_setup.md in the addon repo.
    """
    peer = get_remote_address(request)
    if not _is_trusted_proxy(peer):
        return peer

    settings = get_settings()
    if settings.client_ip_header:
        value = request.headers.get(settings.client_ip_header)
        if value:
            return value.strip()

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        hops = [h.strip() for h in forwarded.split(",") if h.strip()]
        for hop in reversed(hops):
            if not _is_trusted_proxy(hop):
                return hop

    # Trusted peer, no usable forwarding info — e.g. a health check from the
    # proxy itself, or local dev hitting uvicorn directly from loopback.
    return peer


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
