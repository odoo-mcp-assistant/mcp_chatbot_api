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

import logging

# dataclass: auto-generates __init__ and makes the class behave like a data container
from dataclasses import dataclass

# Annotated: lets us attach metadata to a type hint (used to declare Header parameters)
from typing import Annotated

# Header: tells FastAPI to read a value from the HTTP request headers
# HTTPException: raises an HTTP error response (e.g. 401 Unauthorized)
# status: contains HTTP status code constants (e.g. status.HTTP_401_UNAUTHORIZED = 401)
from fastapi import Header, HTTPException, status

# JWTError: raised by python-jose when the token is invalid, expired, or tampered with
# jwt: the library that decodes and verifies JWT tokens
from jose import JWTError, jwt

# get_settings: returns the cached .env config containing jwt_secret, jwt_algorithm, jwt_audience
from .config import get_settings

# create a logger for this file — messages will appear as "app.auth" in logs
_logger = logging.getLogger(__name__)


# Principal is an immutable data object that represents the identified caller for one request
# frozen=True means its fields cannot be changed after creation
@dataclass(frozen=True)
class Principal:
    partner_id: int | None      # the Odoo partner ID — set for logged-in users, None for anonymous
    session_token: str | None   # random string identifying an anonymous session — None for logged-in users
    anonymous: bool             # True if the user is not logged into Odoo

    @property
    def is_authenticated(self) -> bool:
        # True only when we have a verified partner_id (portal login or OTP-verified anonymous)
        # anonymous users who haven't verified via OTP have partner_id = None → returns False
        return self.partner_id is not None


def verify_token(token: str) -> Principal:
    """Decode and validate a JWT. Raises HTTPException(401) on any issue."""
    settings = get_settings()   # get jwt_secret, jwt_algorithm, jwt_audience from .env
    try:
        # decode the JWT: verifies the signature using jwt_secret, checks expiry, checks audience
        # if any check fails, JWTError is raised
        claims = jwt.decode(
            token,
            settings.jwt_secret,      # the shared secret used to verify the signature
            algorithms=[settings.jwt_algorithm],  # e.g. ["HS256"]
            audience=settings.jwt_audience,       # must match the "aud" claim in the token
        )
    except JWTError as exc:
        # token is invalid, expired, tampered with, or has wrong audience
        _logger.warning("auth: token rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,   # return 401 to the browser
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},     # standard header telling the client to send a Bearer token
        ) from exc

    # token is valid — extract the identity claims and return a Principal object
    return Principal(
        partner_id=claims.get("partner_id"),              # int or None
        session_token=claims.get("session_token"),        # str or None
        anonymous=bool(claims.get("anonymous", True)),    # default True if claim is missing
    )


# current_principal is a FastAPI dependency — it is declared with Depends() in route handlers
# FastAPI automatically calls this function for every protected request and injects the result
async def current_principal(
    authorization: Annotated[str | None, Header()] = None,  # reads the "Authorization" HTTP header
) -> Principal:
    # check that the header exists and starts with "Bearer " (case-insensitive)
    # e.g. "Bearer eyJhbGciOiJIUzI1NiJ9...."
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # strip the "Bearer " prefix (7 characters) to get the raw token string
    token = authorization[7:].strip()

    # verify the token and return the Principal — raises 401 if invalid
    return verify_token(token)
