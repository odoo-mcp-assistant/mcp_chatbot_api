"""
Agentic loop — async LLM + tool execution.

Replaces the original Odoo addon's `_async_process_message`. Natively
async — no thread bridge, no `run_coroutine_threadsafe`, no global
singletons to warm up. Just a flat top-to-bottom function that:

  1. Sends the conversation to the LLM with the union of MCP tools +
     local tools (currently just `remember_fact`).
  2. Loops on tool calls up to `cfg.max_tool_rounds`:
       - `remember_fact`              → handled locally, writes to Odoo.
       - `verify_email_otp`           → session_id injected, success result
                                        promotes the in-flight partner_id.
       - tools in AUTH_REQUIRED_TOOLS → partner_id injected, or blocked
                                        with an auth-flow suggestion.
       - anything else                → forwarded to the MCP server.
  3. If the cap is reached, forces a plain-text wrap-up (with no tools
     advertised, because some LLM providers 400 on `tool_choice="none"`).
"""
from __future__ import annotations

import asyncio
import json
import logging

from .llm_client import get_async_openai
from .mcp_client import get_mcp
from .odoo_config import OdooConfig
from .services import fact as fact_svc

_logger = logging.getLogger(__name__)


# Tools the MCP server exposes that must only be callable when we have
# a verified partner_id. The agent injects partner_id automatically.
AUTH_REQUIRED_TOOLS = {
    "get_orders",
    "create_order",
    "confirm_order",
    "cancel_order",
    "get_order_details",
    "get_my_profile",
    "get_invoices",
    "get_invoice_details",
    "get_unpaid_invoices",
}


# Tools implemented inside FastAPI — never forwarded to the MCP server.
LOCAL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": (
                "Save a durable fact about the user. Use only for statements "
                "the user explicitly wants remembered, or durable attributes: "
                "preferences, ecosystem, dislikes, allergies, profession, "
                "lifestyle. Do NOT use for ephemeral context, product mentions, "
                "or inventory data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The fact, as a concise standalone sentence.",
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Short label: preference, ecosystem, dislike, "
                            "health, profession, lifestyle, general."
                        ),
                    },
                },
                "required": ["text"],
            },
        },
    },
]


AUTH_REQUIRED_SUGGESTION = (
    "This action requires a verified identity. "
    "The user can either sign in to their account, "
    "or verify via email using these steps:\n"
    "Step 1: Ask the user for their email address. "
    "Send ONLY this question and STOP. Do not call any tool.\n"
    "Step 2: Once the user replies with their email, "
    "call send_verification_email with that email. "
    "Tell them a code was sent and STOP. "
    "Do not repeat yourself. One short sentence is enough.\n"
    "Step 3: Once the user replies with the 6-digit code, "
    "call verify_email_otp with their email and code.\n"
    "Step 4: Once verified, retry the original action.\n"
    "IMPORTANT: Each step requires a separate user reply. "
    "Do NOT combine steps. Send one short message per step and wait."
)


async def _handle_remember_fact(
    partner_id: int | None, args: dict,
) -> str:
    if not partner_id:
        return json.dumps({
            "success": False,
            "error": "Cannot save facts for anonymous users.",
        })
    text = (args or {}).get("text") or ""
    text = text.strip()
    if not text:
        return json.dumps({"success": False, "error": "text is required"})
    category = ((args or {}).get("category") or "general").strip() or "general"
    fact_id = await fact_svc.save(partner_id, text, category)
    return json.dumps({"success": True, "id": fact_id, "category": category})


def _extract_reply(msg) -> str:
    """Some providers stash the text under `reasoning_content` or `reasoning`
    instead of `content` — fall through them all."""
    return (
        getattr(msg, "content", None)
        or getattr(msg, "reasoning_content", "")
        or getattr(msg, "reasoning", "")
        or ""
    )


async def process_message(
    user_message: str,
    history: list[dict],
    cfg: OdooConfig,
    authenticated_partner_id: int | None = None,
    session_id: int | None = None,
) -> tuple[str, int | None]:
    """
    Run the agentic loop for one user turn.

    Returns `(reply_text, verified_partner_id_or_None)`. The second value
    is non-None only when `verify_email_otp` succeeded during this call —
    the caller uses it to persist the session→partner link.
    """
    mcp = get_mcp()
    tool_schemas = mcp.tool_schemas + LOCAL_TOOL_SCHEMAS
    llm = get_async_openai(cfg.llm.api_key, cfg.llm.base_url)

    conversation: list[dict] = [{"role": "system", "content": cfg.system_prompt}]
    conversation.extend(history)
    conversation.append({"role": "user", "content": user_message})

    verified_partner_id: int | None = None

    for round_num in range(cfg.max_tool_rounds):
        task = asyncio.current_task()
        if task is not None and task.cancelled():
            return (
                "Sorry, the request timed out. Please try again.",
                verified_partner_id,
            )

        response = await llm.chat.completions.create(
            model=cfg.llm.model_name,
            messages=conversation,
            tools=tool_schemas,
            tool_choice="auto",
            temperature=0.7,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return _extract_reply(message), verified_partner_id

        conversation.append(message)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}

            # verify_email_otp: session_id so the MCP server can link
            # the newly-verified partner onto the correct session row.
            if name == "verify_email_otp":
                args["session_id"] = session_id

            # Auth gate: if the tool needs a verified partner, either inject
            # it or return the auth-flow suggestion back to the LLM.
            if name in AUTH_REQUIRED_TOOLS:
                if not authenticated_partner_id:
                    _logger.info(
                        "agent: round %d '%s' blocked (no partner_id)",
                        round_num + 1, name,
                    )
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": json.dumps({
                            "error": "Authentication required",
                            "suggestion": AUTH_REQUIRED_SUGGESTION,
                        }),
                    })
                    continue
                args["partner_id"] = authenticated_partner_id

            _logger.info(
                "agent: round %d → '%s' args=%s", round_num + 1, name, args,
            )

            # Dispatch: local tool or forward to MCP
            if name == "remember_fact":
                result_text = await _handle_remember_fact(
                    authenticated_partner_id, args,
                )
            else:
                try:
                    mcp_result = await mcp.call_tool(name, arguments=args)
                    result_text = str(mcp_result.content)

                    # verify_email_otp success → promote partner_id in-flight
                    # so tools called later in THIS same turn see the verified id.
                    if name == "verify_email_otp":
                        try:
                            raw = (
                                mcp_result.content[0].text
                                if mcp_result.content else "{}"
                            )
                            parsed = json.loads(raw)
                            if parsed.get("success") and parsed.get("partner_id"):
                                authenticated_partner_id = parsed["partner_id"]
                                verified_partner_id = parsed["partner_id"]
                        except Exception as exc:
                            _logger.debug(
                                "agent: verify_email_otp parse failed: %s", exc,
                            )
                except Exception as exc:
                    _logger.exception("agent: tool '%s' failed: %s", name, exc)
                    result_text = json.dumps({"error": str(exc)})

            conversation.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": result_text,
            })

    # Cap reached — force a plain-text wrap-up. We strip tools from the
    # final request entirely; some providers (Groq) 400 on tool_choice="none"
    # when the model still tries to emit a tool call.
    _logger.warning("agent: tool-round cap (%d) reached", cfg.max_tool_rounds)
    conversation.append({
        "role": "user",
        "content": (
            "You have no tools available. Summarise what you have done so far "
            "and respond to the user in plain text only."
        ),
    })
    try:
        final = await llm.chat.completions.create(
            model=cfg.llm.model_name,
            messages=conversation,
            temperature=0.7,
        )
        return _extract_reply(final.choices[0].message), verified_partner_id
    except Exception as exc:
        _logger.warning("agent: fallback completion failed: %s", exc)
        return (
            "I've looked into your request but wasn't able to finish processing. "
            "Could you please try rephrasing or simplifying your question?"
        ), verified_partner_id
