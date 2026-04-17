"""
Conversation history summarisation via the summary LLM.

Pure async — uses AsyncOpenAI directly, never touches Odoo during the
call. The summary text is written back to the session by the caller.
"""
from __future__ import annotations

import logging

from ..llm_client import get_async_openai
from ..odoo_config import LLMConfig

_logger = logging.getLogger(__name__)


SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation summarizer. "
    "Produce a single concise standalone summary that preserves "
    "all important context: key questions asked, decisions made, "
    "products or data mentioned, and the current state of the conversation. "
    "Replace any previous summary entirely — do not append to it. "
    "Preserve exact product names, order references (e.g. S00108), "
    "email addresses, and technical identifiers exactly as they appear — "
    "never paraphrase or rename them. "
    "CRITICAL — staleness rule: prices, stock levels, promotions, and "
    "availability are historical snapshots, not current values. "
    "When they appear, phrase them as 'previously shown', 'earlier quoted', "
    "or 'at the time'. Never state a price or stock level as if it is current. "
    "The summary is context about what the conversation covered, not "
    "authoritative inventory data — fresh values must come from a new tool call. "
    "Be brief but complete."
)


async def summarize(history: list[dict[str, str]], llm: LLMConfig) -> str:
    if not history:
        return ""

    client = get_async_openai(llm.api_key, llm.base_url)
    formatted = "\n".join(
        f"[{m['role'].upper()}]: {m.get('content') or ''}" for m in history
    )

    response = await client.chat.completions.create(
        model=llm.model_name,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Please summarize this conversation history:\n\n{formatted}",
            },
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content or ""
