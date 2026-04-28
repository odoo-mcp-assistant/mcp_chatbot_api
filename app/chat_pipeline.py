"""
Per-request chat orchestration.

The HTTP handler calls `handle_chat(principal, user_message)` and gets
back the assistant reply. This module assembles everything the agent
needs: session resolution, history + summary, identity + facts injection,
then invokes the agentic loop and persists results.

Ported from `mcp_chatbot/controllers/chatbot_controller.py::receive_message`.
"""

import logging

from .agent import process_message, process_message_stream
from .auth import Principal
from .odoo_config import get_odoo_config
from .services import (
    fact as fact_svc,
    identity as identity_svc,
    message as message_svc,
    session as session_svc,
    summary as summary_svc,
)

_logger = logging.getLogger(__name__)


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough 4-chars-per-token estimate."""
    return sum(len((m.get("content") or "")) // 4 for m in messages)


# ─────────────────────────────────────────────────────────────────────────────
# Non-streaming path (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

async def handle_chat(principal: Principal, user_message: str) -> tuple[str, bool]:
    """Process one user turn. Returns (reply, did_summarize_this_turn)."""
    cfg = get_odoo_config()

    # -------------------------------------------------------------------------
    # STEP 1 — Resolve or create the Odoo session for this user.
    # -------------------------------------------------------------------------
    if principal.partner_id:
        sess = await session_svc.get_or_create(partner_id=principal.partner_id)
    elif principal.session_token:
        sess = await session_svc.get_or_create(session_token=principal.session_token)
    else:
        raise ValueError("Principal has neither partner_id nor session_token")

    session_id = sess["id"]
    effective_partner_id = principal.partner_id or sess.get("partner_id")

    # -------------------------------------------------------------------------
    # STEP 2 — Persist the incoming user message to Odoo immediately.
    # -------------------------------------------------------------------------
    await message_svc.create(session_id, "user", user_message)

    # -------------------------------------------------------------------------
    # STEP 3 — Build history and decide whether to summarize.
    # -------------------------------------------------------------------------
    all_history = await session_svc.get_conversation_history(session_id)
    prior_history = all_history[:-1] if all_history else []
    prior_count = len(prior_history)

    unsummarized_count = prior_count - (sess.get("last_summarized_count") or 0)
    unsummarized_messages = (
        prior_history[-unsummarized_count:] if unsummarized_count > 0 else []
    )

    did_summarize = False
    if _estimate_tokens(unsummarized_messages) >= cfg.summary_interval:
        summary_prefix: list[dict] = []
        if sess.get("history_summary"):
            summary_prefix = [{
                "role": "system",
                "content": (
                    "Here is the previous summary for context — produce a NEW "
                    "standalone summary that incorporates both this and the new "
                    "messages below. Do NOT just append to it:\n\n"
                    f"{sess['history_summary']}"
                ),
            }]
        new_summary = await summary_svc.summarize(
            summary_prefix + unsummarized_messages,
            cfg.summary_llm,
        )
        await session_svc.save_summary(session_id, new_summary, prior_count)
        sess = {**sess, "history_summary": new_summary, "last_summarized_count": prior_count}
        unsummarized_messages = []
        did_summarize = True

    # -------------------------------------------------------------------------
    # STEP 4 — Assemble conversation_history sent to the agent.
    # -------------------------------------------------------------------------
    conversation_history: list[dict] = []

    if sess.get("history_summary"):
        conversation_history.append({
            "role": "system",
            "content": f"Summary of the conversation so far: {sess['history_summary']}",
        })

    conversation_history.extend(unsummarized_messages)

    identity_msg = await identity_svc.build_identity_message(
        authenticated_partner_id=principal.partner_id,
        session_partner_id=sess.get("partner_id"),
    )
    conversation_history = [{"role": "system", "content": identity_msg}] + conversation_history

    if effective_partner_id:
        facts = await fact_svc.list_for_partner(effective_partner_id)
        if facts:
            fact_lines = [
                f"- [{(f.get('category') or 'general')}] {f.get('fact_text') or ''}"
                for f in facts
            ]
            fact_block = (
                "KNOWN FACTS ABOUT THIS USER (personal preferences / history — "
                "NOT current inventory or product data; use only to personalise "
                "recommendations and responses):\n" + "\n".join(fact_lines)
            )
            conversation_history = [
                {"role": "system", "content": fact_block}
            ] + conversation_history
            _logger.info(
                "chat: injected %d facts for partner %s",
                len(facts), effective_partner_id,
            )

    # -------------------------------------------------------------------------
    # STEP 5 — Run the agentic loop.
    # -------------------------------------------------------------------------
    try:
        reply, verified_partner_id = await process_message(
            user_message=user_message,
            history=conversation_history,
            cfg=cfg,
            authenticated_partner_id=effective_partner_id,
            session_id=session_id,
        )
    except session_svc.SessionClosed:
        _logger.info(
            "chat: session %s closed during agent run — skipping persistence",
            session_id,
        )
        return "", False
    except Exception as exc:
        _logger.exception("chat: agent pipeline error: %s", exc)
        reply = "Sorry, I encountered an error. Please try again."
        verified_partner_id = None

    # -------------------------------------------------------------------------
    # STEP 6 — Persist the assistant reply to Odoo.
    # -------------------------------------------------------------------------
    await message_svc.create(session_id, "assistant", reply)

    # -------------------------------------------------------------------------
    # STEP 7 — Update last_activity on the session.
    # -------------------------------------------------------------------------
    if not verified_partner_id:
        try:
            await session_svc.touch_activity(session_id)
        except Exception as exc:
            _logger.warning("chat: touch_activity failed: %s", exc)

    return reply, did_summarize


# ─────────────────────────────────────────────────────────────────────────────
# Streaming path
# ─────────────────────────────────────────────────────────────────────────────

async def handle_chat_stream(principal: Principal, user_message: str):
    """
    Process one user turn with true streaming.

    Yields SSE-compatible dicts:
      {"type": "meta",   "summarized": bool}
      {"type": "chunk",  "text": "..."}
      {"type": "done",   "reply": "...", "summarized": bool}
      {"type": "error",  "message": "..."}
    """
    cfg = get_odoo_config()

    # STEP 1 — Session
    if principal.partner_id:
        sess = await session_svc.get_or_create(partner_id=principal.partner_id)
    elif principal.session_token:
        sess = await session_svc.get_or_create(session_token=principal.session_token)
    else:
        raise ValueError("Principal has neither partner_id nor session_token")

    session_id = sess["id"]
    effective_partner_id = principal.partner_id or sess.get("partner_id")

    # STEP 2 — Save user message
    await message_svc.create(session_id, "user", user_message)

    # STEP 3 — History & summarisation
    all_history = await session_svc.get_conversation_history(session_id)
    prior_history = all_history[:-1] if all_history else []
    prior_count = len(prior_history)

    unsummarized_count = prior_count - (sess.get("last_summarized_count") or 0)
    unsummarized_messages = (
        prior_history[-unsummarized_count:] if unsummarized_count > 0 else []
    )

    did_summarize = False
    if _estimate_tokens(unsummarized_messages) >= cfg.summary_interval:
        summary_prefix: list[dict] = []
        if sess.get("history_summary"):
            summary_prefix = [
                {
                    "role": "system",
                    "content": (
                        "Here is the previous summary for context — produce a NEW "
                        "standalone summary that incorporates both this and the new "
                        "messages below. Do NOT just append to it:\n\n"
                        f"{sess['history_summary']}"
                    ),
                }
            ]
        new_summary = await summary_svc.summarize(
            summary_prefix + unsummarized_messages,
            cfg.summary_llm,
        )
        await session_svc.save_summary(session_id, new_summary, prior_count)
        sess = {
            **sess,
            "history_summary": new_summary,
            "last_summarized_count": prior_count,
        }
        unsummarized_messages = []
        did_summarize = True

    yield {"type": "meta", "summarized": did_summarize}

    # STEP 4 — Assemble conversation_history
    conversation_history: list[dict] = []

    if sess.get("history_summary"):
        conversation_history.append(
            {
                "role": "system",
                "content": f"Summary of the conversation so far: {sess['history_summary']}",
            }
        )

    conversation_history.extend(unsummarized_messages)

    identity_msg = await identity_svc.build_identity_message(
        authenticated_partner_id=principal.partner_id,
        session_partner_id=sess.get("partner_id"),
    )
    conversation_history = [
        {"role": "system", "content": identity_msg}
    ] + conversation_history

    if effective_partner_id:
        facts = await fact_svc.list_for_partner(effective_partner_id)
        if facts:
            fact_lines = [
                f"- [{(f.get('category') or 'general')}] {f.get('fact_text') or ''}"
                for f in facts
            ]
            fact_block = (
                "KNOWN FACTS ABOUT THIS USER (personal preferences / history — "
                "NOT current inventory or product data; use only to personalise "
                "recommendations and responses):\n" + "\n".join(fact_lines)
            )
            conversation_history = [
                {"role": "system", "content": fact_block}
            ] + conversation_history
            _logger.info(
                "chat: injected %d facts for partner %s",
                len(facts),
                effective_partner_id,
            )

    # STEP 5 — Run the agentic loop (true streaming)
    full_reply = ""
    verified_partner_id = None

    try:
        async for event in process_message_stream(
            user_message=user_message,
            history=conversation_history,
            cfg=cfg,
            authenticated_partner_id=effective_partner_id,
            session_id=session_id,
        ):
            if event["type"] == "chunk":
                full_reply += event["text"]
                yield event
            elif event["type"] == "done":
                full_reply = event["reply"]
                verified_partner_id = event.get("verified_partner_id")
                # Don't yield done yet — persist first
            elif event["type"] == "error":
                yield event
                return

    except session_svc.SessionClosed:
        _logger.info("chat: session %s closed during agent run", session_id)
        yield {"type": "error", "message": ""}
        return
    except Exception as exc:
        _logger.exception("chat: agent pipeline error: %s", exc)
        yield {
            "type": "error",
            "message": "Sorry, I encountered an error. Please try again.",
        }
        return

    # STEP 6 — Persist assistant reply
    await message_svc.create(session_id, "assistant", full_reply)

    # STEP 7 — Touch activity
    if not verified_partner_id:
        try:
            await session_svc.touch_activity(session_id)
        except Exception as exc:
            _logger.warning("chat: touch_activity failed: %s", exc)

    yield {"type": "done", "reply": full_reply, "summarized": did_summarize}