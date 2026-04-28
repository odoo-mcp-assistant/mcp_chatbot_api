"""
Agentic loop — async LLM + tool execution.
"""

import asyncio
import json
import logging

from .llm_client import get_async_openai
from .mcp_client import get_mcp
from .odoo_config import OdooConfig
from .services import session as session_svc

_logger = logging.getLogger(__name__)


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


def _extract_reply(msg) -> str:
    """Return only the assistant's public content."""
    return getattr(msg, "content", None) or ""


# ─────────────────────────────────────────────────────────────────────────────
# Non-streaming path (backward compatible)
# ─────────────────────────────────────────────────────────────────────────────

async def process_message(
    user_message: str,
    history: list[dict],
    cfg: OdooConfig,
    authenticated_partner_id: int | None = None,
    session_id: int | None = None,
) -> tuple[str, int | None]:
    mcp = get_mcp()
    tool_schemas = mcp.tool_schemas
    llm = get_async_openai(cfg.llm.api_key, cfg.llm.base_url)

    conversation: list[dict] = [
        {"role": "system", "content": cfg.system_prompt}
    ]
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

        if session_id is not None and not await session_svc.is_open(session_id):
            _logger.info(
                "agent: session %s closed mid-run, aborting at round %d",
                session_id, round_num + 1,
            )
            raise session_svc.SessionClosed()

        response = await llm.chat.completions.create(
            model=cfg.llm.model_name,
            messages=conversation,
            tools=tool_schemas,
            tool_choice="auto",
        )
        message = response.choices[0].message

        reasoning = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )
        if reasoning:
            _logger.info(
                "agent: round %d reasoning: %s", round_num + 1, reasoning
            )
        if message.content:
            _logger.info(
                "agent: round %d content: %s", round_num + 1, message.content
            )
        if message.tool_calls:
            _logger.info(
                "agent: round %d tool_calls: %s",
                round_num + 1,
                [tc.function.name for tc in message.tool_calls],
            )

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

            if name == "verify_email_otp":
                args["session_id"] = session_id

            if name in AUTH_REQUIRED_TOOLS:
                if not authenticated_partner_id:
                    _logger.info(
                        "agent: round %d '%s' blocked (no partner_id)",
                        round_num + 1,
                        name,
                    )
                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": name,
                            "content": json.dumps(
                                {
                                    "error": "Authentication required",
                                    "suggestion": AUTH_REQUIRED_SUGGESTION,
                                }
                            ),
                        }
                    )
                    continue
                args["partner_id"] = authenticated_partner_id

            _logger.info(
                "agent: round %d → '%s' args=%s",
                round_num + 1,
                name,
                args,
            )

            try:
                mcp_result = await mcp.call_tool(name, arguments=args)
                result_text = str(mcp_result.content)

                if name == "verify_email_otp":
                    try:
                        raw = (
                            mcp_result.content[0].text
                            if mcp_result.content
                            else "{}"
                        )
                        parsed = json.loads(raw)
                        if parsed.get("success") and parsed.get("partner_id"):
                            authenticated_partner_id = parsed["partner_id"]
                            verified_partner_id = parsed["partner_id"]
                    except Exception as exc:
                        _logger.debug(
                            "agent: verify_email_otp parse failed: %s", exc
                        )
            except Exception as exc:
                _logger.exception("agent: tool '%s' failed: %s", name, exc)
                result_text = json.dumps({"error": str(exc)})

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result_text,
                }
            )

    _logger.warning("agent: tool-round cap (%d) reached", cfg.max_tool_rounds)
    conversation.append(
        {
            "role": "user",
            "content": (
                "You have no tools available. Summarise what you have done so far "
                "and respond to the user in plain text only."
            ),
        }
    )
    try:
        final = await llm.chat.completions.create(
            model=cfg.llm.model_name,
            messages=conversation,
        )
        return _extract_reply(final.choices[0].message), verified_partner_id
    except Exception as exc:
        _logger.warning("agent: fallback completion failed: %s", exc)
        return (
            "I've looked into your request but wasn't able to finish processing. "
            "Could you please try rephrasing or simplifying your question?"
        ), verified_partner_id


# ─────────────────────────────────────────────────────────────────────────────
# Streaming variant — true token-by-token streaming from Kimi to browser
# ─────────────────────────────────────────────────────────────────────────────

async def process_message_stream(
    user_message: str,
    history: list[dict],
    cfg: OdooConfig,
    authenticated_partner_id: int | None = None,
    session_id: int | None = None,
):
    """
    True streaming agent loop.

    Yields:
      {"type": "chunk", "text": "..."}  — live content tokens (never reasoning)
      {"type": "done",  "reply": "...", "verified_partner_id": int|None}
      {"type": "error", "message": "..."}
    """
    mcp = get_mcp()
    tool_schemas = mcp.tool_schemas
    llm = get_async_openai(cfg.llm.api_key, cfg.llm.base_url)

    conversation: list[dict] = [
        {"role": "system", "content": cfg.system_prompt}
    ]
    conversation.extend(history)
    conversation.append({"role": "user", "content": user_message})

    verified_partner_id: int | None = None

    for round_num in range(cfg.max_tool_rounds):
        task = asyncio.current_task()
        if task is not None and task.cancelled():
            yield {
                "type": "error",
                "message": "Sorry, the request timed out. Please try again.",
            }
            return

        if session_id is not None and not await session_svc.is_open(session_id):
            _logger.info(
                "agent: session %s closed mid-run, aborting at round %d",
                session_id,
                round_num + 1,
            )
            raise session_svc.SessionClosed()

        # ── True streaming call ──────────────────────────────────────────
        stream_resp = await llm.chat.completions.create(
            model=cfg.llm.model_name,
            messages=conversation,
            tools=tool_schemas,
            tool_choice="auto",
            stream=True,
        )

        accumulated_content = ""
        accumulated_reasoning = ""
        tool_calls_acc: dict[int, dict] = {}

        async for chunk in stream_resp:
            if task is not None and task.cancelled():
                return

            delta = chunk.choices[0].delta

            # 1. Reasoning — log only, NEVER forward to user
            reasoning = getattr(delta, "reasoning_content", None) or getattr(
                delta, "reasoning", None
            )
            if reasoning:
                accumulated_reasoning += reasoning

            # 2. Content — forward to user immediately
            content = getattr(delta, "content", None)
            if content:
                accumulated_content += content
                yield {"type": "chunk", "text": content}

            # 3. Tool calls — accumulate silently
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc.id,
                            "type": tc.type or "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.type:
                        tool_calls_acc[idx]["type"] = tc.type
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["function"]["arguments"] += tc.function.arguments

        # Log reasoning after the round (debug only — user won't see it)
        if accumulated_reasoning:
            _logger.info(
                "agent: round %d reasoning: %s", round_num + 1, accumulated_reasoning
            )

        # ── No tool calls → final answer ────────────────────────────────
        if not tool_calls_acc:
            yield {
                "type": "done",
                "reply": accumulated_content,
                "verified_partner_id": verified_partner_id,
            }
            return

        # ── Tool calls detected → execute and loop ──────────────────────
        _logger.info(
            "agent: round %d tool_calls: %s",
            round_num + 1,
            [tc["function"]["name"] for tc in tool_calls_acc.values()],
        )

        # Reconstruct assistant message for conversation history.
        # CRITICAL: Preserve reasoning_content for providers that require it
        # (e.g. Moonshot with thinking enabled).  Also, content must be
        # present (even if null) when tool_calls exist.
        assistant_msg: dict = {
            "role": "assistant",
            "content": accumulated_content if accumulated_content else None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in tool_calls_acc.values()
            ],
        }
        if accumulated_reasoning:
            assistant_msg["reasoning_content"] = accumulated_reasoning

        conversation.append(assistant_msg)

        for tool_call in assistant_msg["tool_calls"]:
            name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}

            if name == "verify_email_otp":
                args["session_id"] = session_id

            if name in AUTH_REQUIRED_TOOLS:
                if not authenticated_partner_id:
                    _logger.info(
                        "agent: round %d '%s' blocked (no partner_id)",
                        round_num + 1,
                        name,
                    )
                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": name,
                            "content": json.dumps(
                                {
                                    "error": "Authentication required",
                                    "suggestion": AUTH_REQUIRED_SUGGESTION,
                                }
                            ),
                        }
                    )
                    continue
                args["partner_id"] = authenticated_partner_id

            _logger.info(
                "agent: round %d → '%s' args=%s",
                round_num + 1,
                name,
                args,
            )

            try:
                mcp_result = await mcp.call_tool(name, arguments=args)
                result_text = str(mcp_result.content)

                if name == "verify_email_otp":
                    try:
                        raw = (
                            mcp_result.content[0].text
                            if mcp_result.content
                            else "{}"
                        )
                        parsed = json.loads(raw)
                        if parsed.get("success") and parsed.get("partner_id"):
                            authenticated_partner_id = parsed["partner_id"]
                            verified_partner_id = parsed["partner_id"]
                    except Exception as exc:
                        _logger.debug(
                            "agent: verify_email_otp parse failed: %s", exc
                        )
            except Exception as exc:
                _logger.exception("agent: tool '%s' failed: %s", name, exc)
                result_text = json.dumps({"error": str(exc)})

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": name,
                    "content": result_text,
                }
            )

        # Loop back for next LLM round

    # ── Cap reached ──────────────────────────────────────────────────────
    _logger.warning("agent: tool-round cap (%d) reached", cfg.max_tool_rounds)
    conversation.append(
        {
            "role": "user",
            "content": (
                "You have no tools available. Summarise what you have done so far "
                "and respond to the user in plain text only."
            ),
        }
    )
    try:
        stream_resp = await llm.chat.completions.create(
            model=cfg.llm.model_name,
            messages=conversation,
            stream=True,
        )
        accumulated_content = ""
        async for chunk in stream_resp:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                accumulated_content += content
                yield {"type": "chunk", "text": content}
        yield {
            "type": "done",
            "reply": accumulated_content,
            "verified_partner_id": verified_partner_id,
        }
    except Exception as exc:
        _logger.warning("agent: fallback completion failed: %s", exc)
        yield {
            "type": "error",
            "message": (
                "I've looked into your request but wasn't able to finish processing. "
                "Could you please try rephrasing or simplifying your question?"
            ),
        }