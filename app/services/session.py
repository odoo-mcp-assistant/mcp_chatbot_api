"""
Session CRUD via odoorpc.

Every call wraps a synchronous odoorpc block with `aodoo()`. Return
values are plain dicts (never odoorpc recordsets) so they're safe to
pass around the event loop and across request boundaries.
"""

import logging
from typing import Any

from ..odoo_client import aodoo, get_client

_logger = logging.getLogger(__name__)

# Fields we commonly read on a session. Keep in sync with the Odoo model.
SESSION_FIELDS = [
    "id", "name", "partner_id", "session_token", "state",
    "history_summary", "last_summarized_count", "last_activity",
]


def _normalize(vals: dict[str, Any]) -> dict[str, Any]:
    """Flatten odoorpc's [id, name] Many2one tuples to just the id."""
    pid = vals.get("partner_id")
    if isinstance(pid, (list, tuple)) and pid:
        vals["partner_id_name"] = pid[1]
        vals["partner_id"] = pid[0]
    return vals


async def get_or_create(
    partner_id: int | None = None,
    session_token: str | None = None,
) -> dict[str, Any]:
    """Return an open session for this identifier, creating one if missing."""
    if not partner_id and not session_token:
        raise ValueError("Either partner_id or session_token must be provided")

    def _sync() -> dict[str, Any]:
        odoo = get_client()
        Session = odoo.env["mcp.chatbot.session"]

        if partner_id:
            ids = Session.search(
                [("partner_id", "=", partner_id), ("state", "=", "open")],
                limit=1,
            )
        else:
            ids = Session.search(
                [("session_token", "=", session_token), ("state", "=", "open")],
                limit=1,
            )

        if not ids:
            vals = {"state": "open"}
            if partner_id:
                vals["partner_id"] = partner_id
            else:
                vals["session_token"] = session_token
            new_id = Session.create(vals)
            ids = [new_id]

        return _normalize(Session.browse(ids[0]).read(SESSION_FIELDS)[0])

    return await aodoo(_sync)


async def touch_activity(session_id: int) -> None:
    def _sync() -> None:
        odoo = get_client()
        odoo.env["mcp.chatbot.session"].browse(session_id).touch_activity()
    await aodoo(_sync)


async def get_conversation_history(session_id: int) -> list[dict[str, str]]:
    """Return [{role, content}, ...] sorted chronologically."""
    def _sync() -> list[dict[str, str]]:
        odoo = get_client()
        Msg = odoo.env["mcp.chatbot.message"]
        ids = Msg.search(
            [("session_id", "=", session_id)],
            order="create_date asc, id asc",
        )
        if not ids:
            return []
        return [
            {"role": m["role"], "content": m["content"] or ""}
            for m in Msg.browse(ids).read(["role", "content"])
        ]
    return await aodoo(_sync)


async def save_summary(
    session_id: int, summary: str, last_summarized_count: int,
) -> None:
    def _sync() -> None:
        odoo = get_client()
        odoo.env["mcp.chatbot.session"].browse(session_id).write({
            "history_summary": summary,
            "last_summarized_count": last_summarized_count,
        })
    await aodoo(_sync)


async def close_session(session_id: int) -> None:
    def _sync() -> None:
        odoo = get_client()
        odoo.env["mcp.chatbot.session"].browse(session_id).action_close()
    await aodoo(_sync)


class SessionClosed(Exception):
    """Raised when the agent loop detects the session was closed mid-run."""


async def is_open(session_id: int) -> bool:
    def _sync() -> bool:
        odoo = get_client()
        rec = odoo.env["mcp.chatbot.session"].browse(session_id).read(["state"])
        return bool(rec) and rec[0].get("state") == "open"
    return await aodoo(_sync)


async def save_rating(
    session_id: int, rating: str, feedback: str = "",
    partner_id: int | None = None,
) -> int:
    def _sync() -> int:
        odoo = get_client()
        vals: dict[str, Any] = {
            "session_id": session_id,
            "rating_text": rating,
            "feedback": feedback or "",
        }
        if partner_id:
            vals["partner_id"] = partner_id
        return odoo.env["mcp.chatbot.rating"].create(vals)
    return await aodoo(_sync)


async def lookup_by_token(session_token: str) -> dict | None:
    def _sync() -> dict | None:
        odoo = get_client()
        Session = odoo.env["mcp.chatbot.session"]
        ids = Session.search([("session_token", "=", session_token)], limit=1)
        if not ids:
            return None
        return _normalize(Session.browse(ids[0]).read(SESSION_FIELDS)[0])
    return await aodoo(_sync)


async def lookup_open_by_partner(partner_id: int) -> dict | None:
    def _sync() -> dict | None:
        odoo = get_client()
        Session = odoo.env["mcp.chatbot.session"]
        ids = Session.search(
            [("partner_id", "=", partner_id), ("state", "=", "open")],
            order="create_date desc",
            limit=1,
        )
        if not ids:
            return None
        return _normalize(Session.browse(ids[0]).read(SESSION_FIELDS)[0])
    return await aodoo(_sync)
