"""
Tool-intent classifier — LLM-as-classifier fallback for Kimi K2's
"thought, did not act" bug.

When the agent's LLM emits ``tool_calls=None`` but its ``reasoning_content``
announced a tool call ("let me search…", "I'll fetch…"), Kimi has decoupled
its reasoning from its action head. We classify the reasoning trace by
asking the same Kimi model a binary yes/no question; if it says yes, we
re-issue the original request with a system-message nudge instructing the
model to actually emit the call.

Why LLM-as-classifier (not a trained sklearn model):
  - Kimi reasoning is multilingual (English / French / Arabic / mixed).
    A TF-IDF + LogReg classifier trained on a small CSV does not generalise
    well across languages — the LLM does, natively.
  - No new infra / model hosting.
  - ~50 input + 1 output tokens per call — cheap.
  - Self-improves as the underlying model improves.
  - Triggered ONLY when ``tool_calls`` is empty AND there is reasoning AND
    the per-turn gate (``tool_called_in_turn`` in agent.py) allows it, so
    it's a no-op on the happy path.

Note: Moonshot does NOT accept ``tool_choice="required"``, so the forced
retry uses a strong system-message nudge plus the default
``tool_choice="auto"``.
"""

import logging
from typing import Any

_logger = logging.getLogger(__name__)


_CLASSIFY_SYSTEM = (
    "You are a binary classifier. Reply with exactly one word: yes or no. "
    "No punctuation, no explanation. The reasoning text may be written in "
    "English, French, Arabic, or any mix of these — judge by intent only, "
    "not by the language used."
)

_CLASSIFY_USER_TEMPLATE = (
    "An AI assistant produced the internal reasoning below but did NOT emit "
    "any tool call. Did this reasoning indicate that the assistant intended "
    "to call a tool (search products, fetch orders, retrieve invoices, look "
    "up a profile, paginate, create/confirm/cancel an order, send a "
    "verification email, get a policy document, etc.) which it then failed "
    "to actually invoke?\n\n"
    "Reasoning:\n{reasoning}\n\n"
    "Answer (yes or no):"
)


async def needs_tool_call(llm: Any, reasoning: str | None) -> bool:
    """Ask the LLM whether the given reasoning trace implied a tool call.

    Returns False on empty input or any error — the conservative choice
    (we'd rather skip a fallback retry than burn tokens on a bad call)."""
    if not reasoning or not reasoning.strip():
        return False
    try:
        resp = await llm.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": _CLASSIFY_USER_TEMPLATE.format(
                    reasoning=reasoning.strip(),
                )},
            ],
            temperature=0,
            max_tokens=4,
        )
        verdict = (resp.choices[0].message.content or "").strip().lower()
        is_yes = verdict.startswith("yes")
        _logger.info(
            "intent_classifier: verdict=%r → needs_tool_call=%s",
            verdict, is_yes,
        )
        return is_yes
    except Exception as exc:
        _logger.warning("intent_classifier: classification failed: %s", exc)
        return False


_FORCE_NUDGE = (
    "Your previous reply indicated in its reasoning that a tool call was "
    "needed, but you did not actually emit one. You MUST now issue the tool "
    "call you intended. Reply ONLY with the tool call (no plain text, no "
    "explanation). If your reasoning was about paginating, call the same "
    "tool with the next page using the same other filters."
)


async def force_tool_call(
    llm: Any,
    model: str,
    conversation: list[dict],
    tool_schemas: list[dict],
) -> Any:
    """Re-issue the chat completion with a strong system-message nudge that
    instructs the model to emit the tool call it planned. Moonshot/Kimi
    does not accept ``tool_choice="required"``, so we rely on the nudge
    plus the default ``tool_choice="auto"`` instead."""
    nudged = list(conversation) + [{"role": "system", "content": _FORCE_NUDGE}]
    return await llm.chat.completions.create(
        model=model,
        messages=nudged,
        tools=tool_schemas,
        tool_choice="auto",
    )
