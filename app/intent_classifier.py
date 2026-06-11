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
    "An AI assistant has exactly these tools available:\n{tools}\n\n"
    "The assistant produced the internal reasoning below but did NOT emit any "
    "tool call — it answered in plain text instead. Based ONLY on the tools "
    "listed above, did this reasoning indicate that the assistant intended to "
    "use one of those tools (it named one, or described an action that maps to "
    "one — e.g. confirming an order, searching products, fetching orders or "
    "invoices, looking up a profile, paginating, sending a verification email) "
    "but then failed to actually invoke it?\n\n"
    "Answer yes only if a listed tool should have been called immediately. "
    "Answer no if the reasoning was just formatting a reply, asking the user a "
    "question, or otherwise did not require any of the tools above. "
    "IMPORTANT: if the reasoning decided to FIRST ask the user for missing "
    "information (their email address, a verification code, which product they "
    "want, a quantity, ...) before any tool could run, that plain-text question "
    "was the correct action — answer no.\n\n"
    "Reasoning:\n{reasoning}\n\n"
    "Answer (yes or no):"
)

# Used when the caller passes no tool schemas — keeps the classifier working
# (with a generic example list) rather than rendering an empty tools block.
_GENERIC_TOOLS = (
    "- (tool list unavailable; common tools: search products, fetch orders, "
    "retrieve invoices, look up a profile, confirm/cancel an order, send a "
    "verification email, get a policy document)"
)


def _format_tools(tool_schemas: list[dict] | None) -> str:
    """Render the OpenAI-style tool schemas as a `- name: description` list
    for the classifier prompt. Falls back to a generic hint if none given."""
    lines: list[str] = []
    for s in tool_schemas or []:
        fn = s.get("function", {})
        name = fn.get("name")
        if not name:
            continue
        desc = (fn.get("description") or "").strip()
        first_line = desc.splitlines()[0].strip() if desc else ""
        lines.append(f"- {name}: {first_line}" if first_line else f"- {name}")
    return "\n".join(lines) if lines else _GENERIC_TOOLS


async def needs_tool_call(
    llm: Any,
    reasoning: str | None,
    tool_schemas: list[dict] | None = None,
) -> bool:
    """Ask the LLM whether the given reasoning trace implied a tool call.

    `tool_schemas` (the same OpenAI-style list the agent advertises) is shown
    to the classifier so it can judge against the REAL tool catalog instead of
    a generic example list — this is what lets it tell a dropped `confirm_order`
    from a plain formatting reply.

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
                    tools=_format_tools(tool_schemas),
                    reasoning=reasoning.strip(),
                )},
            ],
            temperature=1,
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
