"""
Agentic loop — async LLM + tool execution.

Replaces the original Odoo addon's `_async_process_message`. Natively
async — no thread bridge, no `run_coroutine_threadsafe`, no global
singletons to warm up. Just a flat top-to-bottom function that:

  1. Sends the conversation to the LLM with the MCP tool catalog.
  2. Loops on tool calls up to `cfg.max_tool_rounds`:
       - `verify_email_otp`           → session_id injected, success result
                                        promotes the in-flight partner_id.
       - tools in AUTH_REQUIRED_TOOLS → partner_id injected, or blocked
                                        with an auth-flow suggestion.
       - anything else                → forwarded to the MCP server.
  3. If the cap is reached, forces a plain-text wrap-up (with no tools
     advertised, because some LLM providers 400 on `tool_choice="none"`).

Fact memory is handled post-session by `services.fact.extract_and_save`;
the agent no longer has a `remember_fact` tool.
"""

import asyncio
import json
import logging

from .intent_classifier import force_tool_call, needs_tool_call
from .llm_client import get_async_openai
from .mcp_client import get_mcp
from .odoo_config import OdooConfig
from .services import session as session_svc

_logger = logging.getLogger(__name__)


# Tools the MCP server exposes that must only be callable when we have
# a verified partner_id. The agent injects partner_id automatically.
AUTH_REQUIRED_TOOLS = {
    "get_orders",
    "create_order",
    "confirm_order",
    "cancel_order",
    "update_order",
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
    """Some providers stash the text under `reasoning_content` or `reasoning`
    instead of `content` — fall through them all."""
    return (
        getattr(msg, "content", None)
        or getattr(msg, "reasoning_content", "")
        or getattr(msg, "reasoning", "")
        or ""
    )
def _mcp_result_to_text(mcp_result) -> str:
    # mcp_result.content is list[TextContent]; str() on it leaks Python repr
    # (TextContent(type='text', text='...')) into the LLM context, which Kimi
    # mis-parses and pattern-completes from training-data Odoo priors.
    parts = getattr(mcp_result, "content", None) or []
    texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
    return "\n".join(texts) if texts else "{}"

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
    tool_schemas = mcp.tool_schemas
    llm = get_async_openai(cfg.llm.api_key, cfg.llm.base_url)

    conversation: list[dict] = [{"role": "system", "content": cfg.system_prompt}]
    conversation.extend(history)
    conversation.append({"role": "user", "content": user_message})

    verified_partner_id: int | None = None

    # Tracks whether a tool has already been called in any previous round of
    # this turn. Used to gate the intent classifier: empirically Kimi only
    # "lies" (reasons about a tool but doesn't emit it) on the FIRST round.
    # Subsequent rounds with no tool_calls are wrap-up/formatting — running
    # the classifier there wastes work and risks false-positive forced
    # retries. If late-round lies start showing up in the logs, remove the
    # flag and let the classifier run on every no-tool-calls round again.
    tool_called_in_turn = False

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
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        if reasoning:
            _logger.info("agent: round %d reasoning: %s", round_num + 1, reasoning)
        if message.content:
            _logger.info("agent: round %d content: %s", round_num + 1, message.content)
        if message.tool_calls:
            _logger.info("agent: round %d tool_calls: %s", round_num + 1, [tc.function.name for tc in message.tool_calls])

        # Fallback: model produced no tool_calls. The trained classifier
        # decides — from the reasoning trace — whether a tool was intended.
        # If yes, re-issue with a system-message nudge forcing the call.
        # Skipped on rounds where a tool has already been called earlier in
        # this turn (those are wrap-up rounds — the classifier would waste
        # cycles and may false-positive on summarisation reasoning).
        if (
            not message.tool_calls
            and reasoning
            and not tool_called_in_turn
            and await needs_tool_call(llm, reasoning)
        ):
            _logger.warning(
                "agent: round %d classifier flagged dropped tool call — "
                "re-issuing with force-call nudge",
                round_num + 1,
            )
            try:
                forced = await force_tool_call(
                    llm, cfg.llm.model_name, conversation, tool_schemas,
                )
                forced_msg = forced.choices[0].message
                if forced_msg.tool_calls:
                    message = forced_msg
                    reasoning = (
                        getattr(message, "reasoning_content", None)
                        or getattr(message, "reasoning", None)
                    )
                    _logger.info(
                        "agent: round %d forced retry succeeded, tool_calls: %s",
                        round_num + 1,
                        [tc.function.name for tc in message.tool_calls],
                    )
                else:
                    _logger.warning(
                        "agent: round %d forced retry still produced no tool_calls",
                        round_num + 1,
                    )
            except Exception as exc:
                _logger.warning(
                    "agent: round %d forced retry failed: %s",
                    round_num + 1, exc,
                )
        # if the message has no tool_calls then return the llm's response 
        if not message.tool_calls:
            return _extract_reply(message), verified_partner_id

        # Tools are about to execute — gate the classifier off for any
        # subsequent rounds in this turn (they are wrap-up rounds).
        tool_called_in_turn = True

        conversation.append(message)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            # Tools accept argments as dict if the llm doesn't emmit a dict as args we retunr empty dict  
            if not isinstance(args, dict):
                args = {}

            # verify_email_otp: session_id so the MCP server can link the newly-verified partner onto the correct session row and modify the last_activity field.
            if name == "verify_email_otp":
                args["session_id"] = session_id

            # Ban the email-verification flow for already-authenticated users:
            # send_verification_email exists only to verify anonymous visitors.
            # If we already have a partner_id, short-circuit it so no code is
            # ever sent and tell the LLM the user is already verified.
            if name == "send_verification_email" and authenticated_partner_id:
                _logger.info(
                    "agent: round %d 'send_verification_email' blocked (already authenticated, partner_id=%s)",
                    round_num + 1, authenticated_partner_id,
                )
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": json.dumps({
                        "error": "Already authenticated",
                        "suggestion": (
                            "The user is already signed in and verified. "
                            "Do NOT ask for their email or send a verification code. "
                            "Proceed with the original request directly."
                        ),
                    }),
                })
                continue

            # Auth gate: if the tool needs a verified partner, either inject it or return the auth-flow suggestion back to the LLM.
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
                    continue # skips the rest of the loop and restarts the current one with the injected instructions  to not block other tool calls that might be idependant 
                # if the user is authenticated then inject the partner_id in the args 
                args["partner_id"] = authenticated_partner_id

            _logger.info(
                "agent: round %d → '%s' args=%s", round_num + 1, name, args,
            )

            try:
                mcp_result = await mcp.call_tool(name, arguments=args)
                result_text = _mcp_result_to_text(mcp_result)

                if name == "verify_email_otp":
                    try:
                        parsed = json.loads(result_text)
                        if parsed.get("success") and parsed.get("partner_id"):
                            authenticated_partner_id = parsed["partner_id"] # needed if the verification is done mid session for later tool calls existing in AUTH_REQUIRED_TOOLS
                            verified_partner_id = parsed["partner_id"] # the function returns it and it's used as signal that the user became verified in this section 
                    except Exception as exc:
                        _logger.debug(
                            "agent: verify_email_otp parse failed: %s", exc
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
        )
        return _extract_reply(final.choices[0].message), verified_partner_id
    except Exception as exc:
        _logger.warning("agent: fallback completion failed: %s", exc)
        return (
            "I've looked into your request but wasn't able to finish processing. "
            "Could you please try rephrasing or simplifying your question?"
        ), verified_partner_id
