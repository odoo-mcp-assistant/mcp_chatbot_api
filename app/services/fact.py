"""
User-fact storage + post-session extraction.

Two responsibilities live here:

1. CRUD via odoorpc — `list_for_partner` reads all facts for a partner
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
from . import message as message_svc

_logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# CRUD
# ──────────────────────────────────────────────────────────────────────


async def list_for_partner(partner_id: int) -> list[dict]:
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

    all_msgs = await message_svc.list_by_session(session_id)
    user_messages = [
        m["content"] for m in all_msgs
        if m.get("role") == "user" and m.get("content")
    ]

    if not user_messages:
        _logger.info("fact-extract: no user messages — nothing to extract")
        _logger.info("===== FACT EXTRACTION END (saved=0) =====")
        return 0

    joined = "\n".join(f"- {m}" for m in user_messages)
    _logger.info("fact-extract: analyzing %d user messages", len(user_messages))

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
        _logger.warning("fact-extract: LLM call failed: %s", exc)
        _logger.info("===== FACT EXTRACTION END (saved=0, llm error) =====")
        return 0

    raw = response.choices[0].message.content or "{}"
    _logger.info("fact-extract: raw LLM output: %s", raw)

    try:
        parsed = json.loads(raw)
        facts = parsed.get("facts") or []
    except json.JSONDecodeError as exc:
        _logger.warning("fact-extract: invalid JSON from LLM: %s", exc)
        _logger.info("===== FACT EXTRACTION END (saved=0, parse error) =====")
        return 0

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
            _logger.warning("fact-extract: failed to save fact %r: %s", text, exc)

    _logger.info("===== FACT EXTRACTION END (saved=%d) =====", saved)
    return saved
