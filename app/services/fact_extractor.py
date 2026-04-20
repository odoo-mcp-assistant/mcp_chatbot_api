"""
Post-session fact extraction.

Runs in the background after a session is closed. Takes ONLY the user
messages (assistant replies are ignored), asks the summary LLM to pick
out durable personal facts, and saves them to Odoo via fact_svc.

Does not run for anonymous sessions — without a partner_id there's
nowhere to save the facts.
"""
import json
import logging

from ..llm_client import get_async_openai
from ..odoo_config import LLMConfig
from . import fact as fact_svc
from . import message as message_svc

_logger = logging.getLogger(__name__)


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
            await fact_svc.save(partner_id, text, category)
            _logger.info("fact-extract: SAVED [%s] %s", category, text)
            saved += 1
        except Exception as exc:
            _logger.warning("fact-extract: failed to save fact %r: %s", text, exc)

    _logger.info("===== FACT EXTRACTION END (saved=%d) =====", saved)
    return saved
