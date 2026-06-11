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
from collections.abc import AsyncGenerator

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


# How long (seconds) the agent waits AFTER a round's stream has ended for the
# tool-intent classifier verdict. The classifier task is started while the
# round is still streaming (the moment the reasoning trace completes), so it
# normally resolves before this timer is even consulted — the bound only bites
# when the provider is slow or overloaded, where skipping the fallback check
# (conservative "no") beats holding the finished reply hostage: the composer
# stays locked until `final` is emitted.
CLASSIFIER_GRACE_S = 3.0


def _mcp_result_to_text(mcp_result) -> str:
    # mcp_result.content is list[TextContent]; str() on it leaks Python repr
    # (TextContent(type='text', text='...')) into the LLM context, which Kimi
    # mis-parses and pattern-completes from training-data Odoo priors.
    parts = getattr(mcp_result, "content", None) or []
    texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
    return "\n".join(texts) if texts else "{}"


async def _stream_completion(llm, **create_kwargs) -> AsyncGenerator[dict, None]:
    """Run one chat completion with stream=True and consume it.

    Yields `{"type": "delta", "content": <token chunk>}` for every content
    fragment as it arrives, then exactly one terminal
    `{"type": "round", "content": str, "reasoning": str, "tool_calls": list[dict]}`
    carrying the fully assembled message.

    Additionally yields one `{"type": "reasoning_done", "reasoning": str}`
    right before the FIRST content delta when the model produced a reasoning
    trace — thinking models finish reasoning before content starts, so at that
    point the trace is complete and the caller can start working with it (the
    tool-intent classifier) concurrently with the rest of the stream.

    Tool calls arrive fragmented across chunks (id/name once, `arguments` in
    string pieces, all keyed by `index`) and are reassembled here into plain
    OpenAI-format dicts — id and name are set from their first non-empty
    fragment, arguments are concatenated. Reasoning deltas
    (`reasoning_content`/`reasoning`) are accumulated but NOT yielded: they
    feed the intent classifier and the empty-content fallback only.
    """
    stream = await llm.chat.completions.create(stream=True, **create_kwargs)

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    # Which delta attribute carried the reasoning trace ("reasoning_content"
    # for Moonshot/Kimi, "reasoning" elsewhere). Thinking providers require
    # the trace echoed back under the SAME field on assistant tool-call
    # messages, so the caller needs the name, not just the text.
    reasoning_field: str | None = None
    tool_calls_by_index: dict[int, dict] = {}

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue

        text = getattr(delta, "content", None)
        if text:
            if reasoning_parts and not content_parts:
                # First content token → the reasoning phase is over; hand the
                # complete trace to the caller before forwarding any content.
                yield {"type": "reasoning_done", "reasoning": "".join(reasoning_parts)}
            content_parts.append(text)
            yield {"type": "delta", "content": text}

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_field = reasoning_field or "reasoning_content"
        else:
            reasoning = getattr(delta, "reasoning", None)
            if reasoning:
                reasoning_field = reasoning_field or "reasoning"
        if reasoning:
            reasoning_parts.append(reasoning)

        for tc in getattr(delta, "tool_calls", None) or []:
            idx = tc.index if tc.index is not None else 0
            entry = tool_calls_by_index.setdefault(idx, {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if tc.id and not entry["id"]:
                entry["id"] = tc.id
            if tc.function:
                if tc.function.name and not entry["function"]["name"]:
                    entry["function"]["name"] = tc.function.name
                if tc.function.arguments:
                    entry["function"]["arguments"] += tc.function.arguments

    yield {
        "type": "round",
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "reasoning_field": reasoning_field or "reasoning_content",
        "tool_calls": [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)],
    }

async def process_message(
    user_message: str,
    history: list[dict],
    cfg: OdooConfig,
    authenticated_partner_id: int | None = None,
    session_id: int | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Run the agentic loop for one user turn as an async generator.

    Yields event dicts the caller forwards to the client and persists:

      - {"type": "delta", "content": str}
            one streamed token chunk of assistant text, live from the LLM.
            Narration emitted alongside tool calls and the final reply both
            arrive this way — the client just appends them in order.
      - {"type": "tool_start"}
            the current round finished streaming and produced tool calls,
            which are about to execute. The client shows a busy indicator
            until the next round's deltas arrive.
      - {"type": "final", "content": str, "verified_partner_id": int | None}
            terminal event. `content` is the authoritative full text of the
            LAST round (it can differ from what was streamed when the reply
            only existed in the reasoning trace). `verified_partner_id` is
            non-None only when `verify_email_otp` succeeded during this call —
            the caller uses it to persist the session→partner link.

    Raises `session_svc.SessionClosed` if the session is closed mid-run.
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
        if session_id is not None and not await session_svc.is_open(session_id):
            _logger.info(
                "agent: session %s closed mid-run, aborting at round %d",
                session_id, round_num + 1,
            )
            raise session_svc.SessionClosed()

        # Stream this round's completion: content tokens are forwarded to the
        # client the moment they arrive; the terminal "round" event carries the
        # fully assembled message (content + reasoning + reassembled tool calls)
        # that the loop logic below works with.
        round_msg: dict = {}
        classifier_task: asyncio.Task | None = None
        async for ev in _stream_completion(
            llm,
            model=cfg.llm.model_name,
            messages=conversation,
            tools=tool_schemas,
            tool_choice="auto",
        ):
            if ev["type"] == "delta":
                yield ev
            elif ev["type"] == "reasoning_done":
                # The reasoning trace is complete while content is still
                # streaming — start the tool-intent classifier NOW so its
                # verdict overlaps the stream instead of stalling the turn
                # after the user has already watched the reply finish.
                # Speculative: cancelled below if the round turns out to have
                # tool calls (where the classifier is irrelevant).
                if not tool_called_in_turn and ev.get("reasoning"):
                    classifier_task = asyncio.create_task(
                        needs_tool_call(llm, ev["reasoning"], tool_schemas)
                    )
            else:
                round_msg = ev

        content = round_msg.get("content") or ""
        reasoning = round_msg.get("reasoning") or ""
        reasoning_field = round_msg.get("reasoning_field") or "reasoning_content"
        tool_calls = round_msg.get("tool_calls") or []

        if reasoning:
            _logger.info("agent: round %d reasoning: %s", round_num + 1, reasoning)
        if content:
            _logger.info("agent: round %d content: %s", round_num + 1, content)
        if tool_calls:
            _logger.info("agent: round %d tool_calls: %s", round_num + 1, [tc["function"]["name"] for tc in tool_calls])

        # Fallback: model produced no tool_calls. The trained classifier
        # decides — from the reasoning trace — whether a tool was intended.
        # If yes, re-issue with a system-message nudge forcing the call.
        # Skipped on rounds where a tool has already been called earlier in
        # this turn (those are wrap-up rounds — the classifier would waste
        # cycles and may false-positive on summarisation reasoning).
        # Resolve the tool-intent classifier verdict. The task was usually
        # started mid-stream (reasoning_done) and has been running while the
        # content painted, so this await is normally instant; the grace bound
        # covers tasks still in flight — and the rare round that produced
        # reasoning but no content (no reasoning_done fired), where the
        # classifier only gets started here.
        flagged_dropped_call = False
        if not tool_calls and reasoning and not tool_called_in_turn:
            if classifier_task is None:
                classifier_task = asyncio.create_task(
                    needs_tool_call(llm, reasoning, tool_schemas)
                )
            try:
                flagged_dropped_call = await asyncio.wait_for(
                    classifier_task, CLASSIFIER_GRACE_S
                )
            except asyncio.TimeoutError:
                # wait_for cancelled the task; conservative "no" — skipping a
                # fallback retry beats stalling the already-painted reply.
                _logger.warning(
                    "agent: round %d intent classifier exceeded %.0fs grace — skipping",
                    round_num + 1, CLASSIFIER_GRACE_S,
                )
        elif classifier_task is not None:
            # Round produced tool calls after all — the speculative check is
            # irrelevant; drop it without waiting.
            classifier_task.cancel()

        if flagged_dropped_call:
            _logger.warning(
                "agent: round %d classifier flagged dropped tool call — "
                "re-issuing with force-call nudge",
                round_num + 1,
            )
            try:
                # The forced retry is NOT streamed: this path fires when the
                # model produced reasoning but no tool call, so there is almost
                # never user-visible content to stream — and any content the
                # original round DID stream is superseded by the final event's
                # authoritative rebuild on the client.
                forced = await force_tool_call(
                    llm, cfg.llm.model_name, conversation, tool_schemas,
                )
                forced_msg = forced.choices[0].message
                if forced_msg.tool_calls:
                    # Convert the SDK message to the same plain-dict shape the
                    # streamed path produces, so the code below sees one format.
                    content = forced_msg.content or ""
                    if getattr(forced_msg, "reasoning_content", None):
                        reasoning = forced_msg.reasoning_content
                        reasoning_field = "reasoning_content"
                    elif getattr(forced_msg, "reasoning", None):
                        reasoning = forced_msg.reasoning
                        reasoning_field = "reasoning"
                    else:
                        reasoning = ""
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "",
                            },
                        }
                        for tc in forced_msg.tool_calls
                    ]
                    _logger.info(
                        "agent: round %d forced retry succeeded, tool_calls: %s",
                        round_num + 1,
                        [tc["function"]["name"] for tc in tool_calls],
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
        # No tool calls → the streamed content WAS the final reply. The final
        # event re-carries the full text (falling back to the reasoning trace
        # for providers that stash the reply there and stream no content).
        if not tool_calls:
            yield {
                "type": "final",
                "content": content.strip() or reasoning.strip() or "",
                "verified_partner_id": verified_partner_id,
            }
            return

        # Tools are about to execute — gate the classifier off for any
        # subsequent rounds in this turn (those are wrap-up rounds) and tell
        # the client to show its busy indicator until the next round streams.
        tool_called_in_turn = True
        yield {"type": "tool_start"}

        # The streamed path has no SDK message object to append — rebuild the
        # assistant message in OpenAI dict format (content may be empty when
        # the model went straight to tool calls).
        assistant_msg: dict = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": tool_calls,
        }
        # Thinking providers (Moonshot/Kimi) reject the next request with 400
        # "thinking is enabled but reasoning_content is missing" unless the
        # reasoning trace is echoed back on assistant tool-call messages. The
        # old non-streamed code got this for free by appending the SDK object;
        # the rebuilt dict must carry it explicitly, under the same field name
        # the provider streamed it with.
        if reasoning:
            assistant_msg[reasoning_field] = reasoning
        conversation.append(assistant_msg)

        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"] or "{}")
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
                    "tool_call_id": tool_call["id"],
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
                        "tool_call_id": tool_call["id"],
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
                "tool_call_id": tool_call["id"],
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
        # Streamed like every other round — the wrap-up text types out live.
        round_msg = {}
        async for ev in _stream_completion(
            llm,
            model=cfg.llm.model_name,
            messages=conversation,
        ):
            if ev["type"] == "delta":
                yield ev
            else:
                round_msg = ev
        yield {
            "type": "final",
            "content": (
                (round_msg.get("content") or "").strip()
                or (round_msg.get("reasoning") or "").strip()
                or ""
            ),
            "verified_partner_id": verified_partner_id,
        }
    except Exception as exc:
        _logger.warning("agent: fallback completion failed: %s", exc)
        yield {
            "type": "final",
            "content": (
                "I've looked into your request but wasn't able to finish processing. "
                "Could you please try rephrasing or simplifying your question?"
            ),
            "verified_partner_id": verified_partner_id,
        }
