"""
User-fact storage + post-session extraction.

Two responsibilities live here:

1. CRUD via odoorpc — `list_facts_for_partner` reads all facts for a partner
   (newest first, no ranking) so the chat pipeline can inject them into
   the system prompt; `save` writes one fact row.

2. Post-session extraction — `extract_and_save` is scheduled as a
   FastAPI background task from /close. It takes ONLY the user messages
   of the closed session, asks the summary LLM to pick out durable
   personal facts, and persists them via `save`. Skipped for anonymous
   sessions (no partner_id → nowhere to save).
"""

import json
import logging

from ..llm_client import get_async_openai
from ..odoo_client import get_client
from ..odoo_config import LLMConfig
from . import session as session_svc

_logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# CRUD
# ──────────────────────────────────────────────────────────────────────


async def list_facts_for_partner(partner_id: int) -> list[dict]:
    """Newest first. Returns [{id, fact_text, category}, ...]."""
    odoo = get_client()
    Fact = odoo.env["mcp.chatbot.user.fact"]
    ids = Fact.search([("partner_id", "=", partner_id)], order="create_date desc")
    if not ids:
        return []
    return Fact.browse(ids).read(["id", "fact_text", "category"])


async def save(partner_id: int, fact_text: str, category: str = "general") -> int:
    odoo = get_client()
    return odoo.env["mcp.chatbot.user.fact"].create({
        "partner_id": partner_id,
        "fact_text": fact_text,
        "category": category,
    })


# ──────────────────────────────────────────────────────────────────────
# Post-session extraction
# ──────────────────────────────────────────────────────────────────────


FACT_EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable personal facts from a user's chat messages. "
    "Consider ONLY statements the user made about themselves: "
    "preferences, ecosystem (devices/tools they own), dislikes, allergies, "
    "profession, lifestyle, language, location. "
    "IGNORE: one-off product mentions, things they were just browsing, "
    "polite conversation, and questions they asked the bot. "
    "\n\n"
    "Return STRICT JSON — no prose, no markdown, no explanation. Schema: "
    '{"facts": [{"text": "<concise standalone sentence>", '
    '"category": "preference|ecosystem|dislike|health|profession|lifestyle|general"}]}'
    "\n\n"
    'If nothing is worth remembering, return {"facts": []}.'
)

# This function is called in the close_session function on calling the endpoint /close as background task to not block the response sent to the front 
async def extract_and_save(
    session_id: int,
    partner_id: int,
    llm: LLMConfig,
) -> int:
    """Run extraction on a closed session. Returns the number of facts saved."""
    _logger.info(
        "===== FACT EXTRACTION START (session=%d, partner=%d) =====",
        session_id, partner_id,
    )

    # Pull every message from the closed session, then keep only what the user
    # actually said. Assistant replies are noise for fact extraction — we only
    # want first-person statements that could be durable facts.
    all_msgs = await session_svc.get_conversation_history(session_id)
    user_messages = [
        m["content"] for m in all_msgs
        if m.get("role") == "user" and m.get("content")
    ]

    # Empty session (e.g. user opened the chat but never typed) — bail out
    # before spending an LLM call.
    if not user_messages:
        _logger.info("fact-extract: no user messages — nothing to extract")
        _logger.info("===== FACT EXTRACTION END (saved=0) =====")
        return 0

    # Format the user turns as a bulleted list so the extraction LLM sees one
    # statement per line — easier for it to reason about than a wall of text.
    joined = "\n".join(f"- {m}" for m in user_messages)
    _logger.info("fact-extract: analyzing %d user messages", len(user_messages))

    # Ask the summary LLM to pick durable facts out of the user's messages.
    # Low temperature + JSON response format because we want deterministic,
    # parseable output — not creative prose. The system prompt defines the
    # exact schema ({"facts": [{"text", "category"}]}).
    client = get_async_openai(llm.api_key, llm.base_url)
    try:
        response = await client.chat.completions.create(
            model=llm.model_name,
            messages=[
                {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"User messages from the conversation:\n{joined}"},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        # Network/LLM failure is non-fatal — extraction is best-effort and
        # runs as a background task, so we swallow and return 0.
        _logger.warning("fact-extract: LLM call failed: %s", exc)
        _logger.info("===== FACT EXTRACTION END (saved=0, llm error) =====")
        return 0

    raw = response.choices[0].message.content or "{}"
    _logger.info("fact-extract: raw LLM output: %s", raw)

    # Parse the JSON envelope. Even with response_format=json_object the model
    # can occasionally return malformed output, so guard the parse and treat
    # a parse failure the same as "no facts found".
    try:
        parsed = json.loads(raw)
        facts = parsed.get("facts") or []
    except json.JSONDecodeError as exc:
        _logger.warning("fact-extract: invalid JSON from LLM: %s", exc)
        _logger.info("===== FACT EXTRACTION END (saved=0, parse error) =====")
        return 0

    # Persist each extracted fact one row at a time. We validate each item
    # defensively (must be a dict with non-empty text) because the LLM output
    # is untrusted — and fall back to "general" when category is missing or
    # blank to satisfy the Odoo model's selection field.
    saved = 0
    for item in facts:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        category = (item.get("category") or "general").strip() or "general"
        if not text:
            continue
        try:
            await save(partner_id, text, category)
            _logger.info("fact-extract: SAVED [%s] %s", category, text)
            saved += 1
        except Exception as exc:
            # One bad row shouldn't kill the rest — keep going so we save
            # whatever facts we can.
            _logger.warning("fact-extract: failed to save fact %r: %s", text, exc)

    _logger.info("===== FACT EXTRACTION END (saved=%d) =====", saved)
    return saved
