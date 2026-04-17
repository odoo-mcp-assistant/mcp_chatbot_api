"""
JWT authentication.

Odoo is the token **issuer** — when the widget loads, it calls an Odoo
endpoint that mints a short-lived JWT containing the caller's identity
(partner_id if logged in, session_token if anonymous). The shared secret
lives in `.env` (`JWT_SECRET`) and must match the value in Odoo's
`ir.config_parameter['mcp_chatbot.jwt_secret']`.

FastAPI is the **verifier** — every protected endpoint depends on
`current_principal`, which decodes the token and returns a Principal
describing who the caller is. Invalid / expired / wrong-audience tokens
produce a 401.

Token claims
------------
    {
      "sub":            "<partner_id>" | "anon:<session_token>",
      "partner_id":     int | null,
      "session_token":  str | null,
      "anonymous":      bool,
      "exp":            <unix-ts>,
      "iat":            <unix-ts>,
      "aud":            "mcp-chatbot-api"
    }
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt

from .config import get_settings

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Principal:
    """The identified caller for one request."""
    partner_id: int | None
    session_token: str | None
    anonymous: bool

    @property
    def is_authenticated(self) -> bool:
        """True when we have a verified partner (portal login or OTP)."""
        return self.partner_id is not None


def verify_token(token: str) -> Principal:
    """Decode and validate a JWT. Raises HTTPException(401) on any issue."""
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
        )
    except JWTError as exc:
        _logger.warning("auth: token rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return Principal(
        partner_id=claims.get("partner_id"),
        session_token=claims.get("session_token"),
        anonymous=bool(claims.get("anonymous", True)),
    )


async def current_principal(
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """FastAPI dependency — extracts and verifies the Bearer token."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[7:].strip()
    return verify_token(token)
