"""
Session CRUD via odoorpc.

Return values are plain dicts (never odoorpc recordsets) so they're safe
to pass around and across request boundaries.

The file is split into two groups:
  - functions used by the chat pipeline (chat_pipeline.py + agent.py)
  - functions used by the chat router (routers/chat.py)
A small shared section at the top holds the constants and helpers that
both groups depend on.
"""

# Any: used as the value type in dict[str, Any] because Odoo records mix
# strings, ints, lists (Many2one), and bools — there's no single type for them.
from typing import Any

# get_client(): returns the shared, already-authenticated odoorpc connection
# opened once at startup in odoo_client.py
from ..odoo_client import get_client


# ======================================================================
# SHARED — used by both groups below
# ======================================================================

# Default field set we read whenever we fetch a session record.
# Listing them explicitly avoids reading every field on the model (faster + smaller payload).
# Keep in sync with the Odoo model `mcp.chatbot.session` if fields are added/renamed.
SESSION_FIELDS = [
    "id", "name", "partner_id", "session_token", "state",
    "history_summary", "last_summarized_count", "last_activity",
]


def _normalize(vals: dict[str, Any]) -> dict[str, Any]:
    """Flatten odoorpc's [id, name] Many2one tuples to just the id."""
    # Odoo returns Many2one fields as [id, "display_name"] — odoorpc preserves that shape.
    # Downstream code only ever needs the id, so we collapse the tuple here.
    pid = vals.get("partner_id")
    # Defensive: only flatten if it actually IS a list/tuple (an int means it was already flattened
    # by an earlier call, and None means anonymous → leave untouched).
    if isinstance(pid, (list, tuple)) and pid:
        # keep the human-readable name on a side key in case some caller wants to display it
        vals["partner_id_name"] = pid[1]
        # replace the tuple with just the id so vals["partner_id"] is now a plain int
        vals["partner_id"] = pid[0]
    return vals







# ======================================================================
# CHAT PIPELINE — called from chat_pipeline.py and agent.py
# (per-message session resolution, history loading, summary persistence,
#  liveness checks, and the SessionClosed signal)
# ======================================================================


# Called by chat_pipeline at the start of every /message request to resolve the session
# for this caller. Either reuses the open session or creates a fresh one.
async def get_or_create(
    partner_id: int | None = None,        # set for authenticated callers
    session_token: str | None = None,     # set for anonymous callers (uuid kept in sessionStorage)
) -> dict[str, Any]:
    """Return an open session for this identifier, creating one if missing."""
    # Programmer error guard — handler in chat.py should already reject this case before we get here.
    if not partner_id and not session_token:
        raise ValueError("Either partner_id or session_token must be provided")

    odoo = get_client()
    # Session is the odoorpc proxy for the Odoo model mcp.chatbot.session;
    # calling search/create/browse/read on it issues XML-RPC calls under the hood.
    Session = odoo.env["mcp.chatbot.session"]

    # Look for an OPEN session belonging to this caller. The two branches use a different
    # identifier (partner_id vs session_token) but the rest of the logic is identical.
    if partner_id:
        ids = Session.search(
            # Odoo domain = list of (field, operator, value) tuples; ANDed together by default
            [("partner_id", "=", partner_id), ("state", "=", "open")],
            limit=1,    # we only ever expect one open session per identity
        )
    else:
        ids = Session.search(
            [("session_token", "=", session_token), ("state", "=", "open")],
            limit=1,
        )

    # No open session → create one. We pass either partner_id OR session_token (never both)
    # so the row is owned by exactly one identity.
    if not ids:
        vals = {"state": "open"}
        if partner_id:
            vals["partner_id"] = partner_id
        else:
            vals["session_token"] = session_token
        # create() returns the new record's id
        new_id = Session.create(vals)
        ids = [new_id]

    # browse(ids[0]) → recordset wrapping that one id
    # .read(SESSION_FIELDS) → list of dicts (one per id, here just one)
    # [0] → the single dict
    # _normalize() → flatten partner_id tuple
    return _normalize(Session.browse(ids[0]).read(SESSION_FIELDS)[0])







# Used by the chat pipeline to load past messages when assembling the LLM context.
# Order is strictly chronological so the assistant sees the conversation in the order it happened.
async def get_conversation_history(session_id: int) -> list[dict[str, str]]:
    """Return [{role, content}, ...] sorted chronologically."""
    odoo = get_client()
    Msg = odoo.env["mcp.chatbot.message"]
    # search() returns just the ids; we apply the ordering at the DB level
    # rather than sorting client-side. The "id asc" tiebreaker keeps order stable when
    # two messages share the exact same create_date (rare but possible at high throughput).
    ids = Msg.search(
        [("session_id", "=", session_id)],
        order="create_date asc, id asc",
    )
    # No messages yet (brand-new session) → return [] rather than calling read() on empty ids.
    if not ids:
        return []
    # browse + read in one shot fetches all messages with one XML-RPC round trip.
    # `m["content"] or ""` guards against NULL/False content fields (Odoo stores empty as False).
    return [
        {"role": m["role"], "content": m["content"] or ""}
        for m in Msg.browse(ids).read(["role", "content"])
    ]








# Persists the rolling history summary produced by services/summary.py.
# `last_summarized_count` is the count of raw messages folded into the summary so far —
# the next summarisation pass picks up from there instead of re-summarising old material.
async def save_summary(
    session_id: int, summary: str, last_summarized_count: int,
) -> None:
    odoo = get_client()
    # write({...}) updates only the fields in the dict, leaves all other fields untouched.
    odoo.env["mcp.chatbot.session"].browse(session_id).write({
        "history_summary": summary,
        "last_summarized_count": last_summarized_count,
    })








# Bumps the last_activity timestamp on the session so the idle-timeout sweeper
# in Odoo doesn't auto-close active sessions. The Odoo-side method is in mcp.chatbot.session.
async def touch_activity(session_id: int) -> None:
    odoo = get_client()
    # browse(id) builds a recordset from a known id (no DB query yet);
    # then we call the custom Odoo method touch_activity() on it.
    odoo.env["mcp.chatbot.session"].browse(session_id).touch_activity()








# Quick liveness check used by the agent loop between tool rounds.
async def is_open(session_id: int) -> bool:
    odoo = get_client()
    # Read only the `state` field — minimal payload, no need for the full session.
    rec = odoo.env["mcp.chatbot.session"].browse(session_id).read(["state"])
    # `bool(rec)` → False if the record was deleted (read returns []).
    # If the session vanished, treat it as not-open.
    return bool(rec) and rec[0].get("state") == "open"






# Custom exception raised by the agent loop when, between two LLM rounds, it discovers
# the user closed the chat (via /close on another tab, or via Odoo admin). Lets the loop
# bail out cleanly instead of trying to persist a reply to a closed session.
class SessionClosed(Exception):
    """Raised when the agent loop detects the session was closed mid-run."""












# ======================================================================
# CHAT ROUTER — called from routers/chat.py
# (lookups for /history and /close, rating persistence, session closing)
# ======================================================================


# Used by /history and /close for AUTHENTICATED callers: only returns OPEN sessions.
# We pick the most recent one (in case of historical leftovers) — there should normally
# only be one open session per partner, but ordering desc + limit=1 makes it deterministic.
async def lookup_open_by_partner(partner_id: int) -> dict | None:
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








# Used by /history and /close for ANONYMOUS callers: the only thing identifying them is
# the uuid in their sessionStorage. Returns the session regardless of state (open OR closed)
# so the frontend can decide whether to start a new one.
async def lookup_by_token(session_token: str) -> dict | None:
    odoo = get_client()
    Session = odoo.env["mcp.chatbot.session"]
    ids = Session.search([("session_token", "=", session_token)], limit=1)
    if not ids:
        # No session attached to this token → caller will get status="not_found".
        return None
    return _normalize(Session.browse(ids[0]).read(SESSION_FIELDS)[0])












# Persists a user rating + optional feedback as a separate mcp.chatbot.rating record
# (linked to the session by Many2one). Ratings are a distinct model so they survive even if
# the session is later archived.
async def save_rating(
    session_id: int, rating: str, feedback: str = "",
    partner_id: int | None = None,
) -> int:
    odoo = get_client()
    vals: dict[str, Any] = {
        "session_id": session_id,
        "rating_text": rating,
        "feedback": feedback or "",   # never write False/None — the field is a string
    }
    # Only attach a partner_id for authenticated raters; anonymous ratings stay un-linked.
    if partner_id:
        vals["partner_id"] = partner_id
    # Returns the new rating record's id (used by the caller for logging only).
    return odoo.env["mcp.chatbot.rating"].create(vals)







# Marks the session as closed via the model's action_close() method
# (rather than write({"state":"closed"}) so any Odoo-side hooks/automations still fire).
async def close_session(session_id: int) -> None:
    odoo = get_client()
    odoo.env["mcp.chatbot.session"].browse(session_id).action_close()




# ======================================================================
# CONVERSATIONS SIDEBAR — called from routers/chat.py
# (read-only browsing of a logged-in user's past conversations)
# ======================================================================

# Fields we read for each row of the sidebar. `message_count` and
# `session_rating_text` are computed Odoo fields — reading them triggers
# the compute, which is fine for the modest per-user session counts here.
CONVERSATION_LIST_FIELDS = [
    "id", "create_date", "state", "message_count", "session_rating_text",
]

# Longest preview title we keep; the rest is trimmed with an ellipsis.
_TITLE_MAX_LEN = 60


def _iso_utc(value: Any) -> str:
    """Turn odoorpc's naive UTC datetime string into an ISO-8601 'Z' string.

    odoorpc hands datetimes back as "YYYY-MM-DD HH:MM:SS" in UTC. Appending
    'Z' (and swapping the space for 'T') lets the browser parse it as UTC and
    render it in the visitor's local timezone.
    """
    if not value:
        return ""
    return str(value).replace(" ", "T") + "Z"


def _first_user_message_previews(session_ids: list[int]) -> dict[int, str]:
    """Map session_id → trimmed text of that session's FIRST user message.

    One batched search/read across all the caller's sessions (instead of one
    query per session) so the sidebar costs two round trips total. Sessions
    with no user message yet are simply absent from the map.
    """
    odoo = get_client()
    Msg = odoo.env["mcp.chatbot.message"]
    mids = Msg.search(
        [("session_id", "in", session_ids), ("role", "=", "user")],
        order="create_date asc, id asc",
    )
    if not mids:
        return {}

    previews: dict[int, str] = {}
    for m in Msg.browse(mids).read(["session_id", "content"]):
        sid = m["session_id"]
        # Many2one comes back as [id, name]; collapse to the id.
        if isinstance(sid, (list, tuple)) and sid:
            sid = sid[0]
        # Keep only the earliest user message per session (first one wins
        # because the search is ordered chronologically).
        if sid in previews:
            continue
        text = (m.get("content") or "").strip()
        if not text:
            continue
        if len(text) > _TITLE_MAX_LEN:
            text = text[:_TITLE_MAX_LEN].rstrip() + "…"
        previews[sid] = text
    return previews


# Lists every session owned by this partner, newest first. Used by
# GET /mcp_chatbot/conversations to populate the sidebar.
async def list_by_partner(partner_id: int) -> list[dict[str, Any]]:
    odoo = get_client()
    Session = odoo.env["mcp.chatbot.session"]
    ids = Session.search(
        [("partner_id", "=", partner_id)],
        order="create_date desc",
    )
    if not ids:
        return []

    rows = Session.browse(ids).read(CONVERSATION_LIST_FIELDS)
    previews = _first_user_message_previews(ids)

    conversations: list[dict[str, Any]] = []
    for row in rows:
        sid = row["id"]
        conversations.append({
            "id": sid,
            "title": previews.get(sid) or "New conversation",
            "created_at": _iso_utc(row.get("create_date")),
            "message_count": row.get("message_count") or 0,
            "state": row.get("state") or "open",
            "rating": row.get("session_rating_text") or "none",
        })
    return conversations


# Returns the chronological messages of one session, but ONLY if that session
# belongs to `partner_id`. Returns None when the session does not exist or is
# owned by someone else — the router maps that to a 404 so a logged-in user
# can never read another partner's conversation by guessing ids.
async def get_owned_history(
    session_id: int, partner_id: int,
) -> tuple[str, list[dict[str, str]]] | None:
    odoo = get_client()
    Session = odoo.env["mcp.chatbot.session"]
    rec = Session.browse(session_id).read(["partner_id", "state"])
    if not rec:
        return None

    owner = rec[0].get("partner_id")
    if isinstance(owner, (list, tuple)) and owner:
        owner = owner[0]
    if owner != partner_id:
        return None

    state = rec[0].get("state") or "closed"
    messages = await get_conversation_history(session_id)
    return state, messages
