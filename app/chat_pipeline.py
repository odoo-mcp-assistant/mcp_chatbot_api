"""
Per-request chat orchestration.

The HTTP handler calls `handle_chat(principal, user_message)` and gets
back the assistant reply. This module assembles everything the agent
needs: session resolution, history + summary, identity + facts injection,
then invokes the agentic loop and persists results.

Ported from `mcp_chatbot/controllers/chatbot_controller.py::receive_message`.
"""

import logging

from .agent import process_message
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


async def handle_chat(principal: Principal, user_message: str) -> tuple[str, bool]:
    """Process one user turn. Returns (reply, did_summarize_this_turn)."""
    cfg = get_odoo_config()

    # -------------------------------------------------------------------------
    # STEP 1 — Resolve or create the Odoo session for this user.
    #
    # A Principal is either an authenticated Odoo contact (has partner_id) or
    # an anonymous visitor identified only by a session_token (e.g. a widget
    # user who hasn't logged in yet). Either way we get back a session dict.
    # -------------------------------------------------------------------------
    if principal.partner_id:
        sess = await session_svc.get_or_create(partner_id=principal.partner_id)
    elif principal.session_token:
        sess = await session_svc.get_or_create(session_token=principal.session_token)
    else:
        raise ValueError("Principal has neither partner_id nor session_token")

    session_id = sess["id"]

    # An anonymous session can become "linked" to a partner mid-session after
    # the user verifies their email via OTP. In that case sess["partner_id"] is
    # set even though principal.partner_id is None, so we merge both sources.
    effective_partner_id = principal.partner_id or sess.get("partner_id")

    # -------------------------------------------------------------------------
    # STEP 2 — Persist the incoming user message to Odoo immediately.
    #
    # We save it before doing any LLM work so it's never lost if the pipeline
    # crashes later. It is saved with role="user".
    # -------------------------------------------------------------------------
    await message_svc.create(session_id, "user", user_message)

    # -------------------------------------------------------------------------
    # STEP 3 — Build history and decide whether to summarize.
    #
    # get_conversation_history() returns ALL messages for this session from
    # the DB, including the user message we just saved. We drop the last one
    # with [:-1] because the agent receives the current message separately via
    # the `user_message` argument — including it twice would confuse the model.
    #
    #   DB:           [msg1, msg2, msg3, msg4, <current user msg>]
    #   all_history:  [msg1, msg2, msg3, msg4, <current user msg>]
    #   prior_history:[msg1, msg2, msg3, msg4]   ← what we work with
    # -------------------------------------------------------------------------
    all_history = await session_svc.get_conversation_history(session_id)
    prior_history = all_history[:-1] if all_history else []
    prior_count = len(prior_history)

    # Figure out which messages have NOT been summarized yet.
    # last_summarized_count is how many messages were already folded into
    # history_summary the last time we ran summarization.
    #
    #   prior_count=4, last_summarized_count=2  →  unsummarized_count=2
    #   unsummarized_messages = [msg3, msg4]    ← only the tail
    unsummarized_count = prior_count - (sess.get("last_summarized_count") or 0)
    unsummarized_messages = (
        prior_history[-unsummarized_count:] if unsummarized_count > 0 else []
    )

    # If the unsummarized tail is large enough, compress it into a summary.
    # The LLM is asked to produce a single NEW summary that merges the old
    # summary (if any) with the new messages — not just append to it.
    # After this block, unsummarized_messages is cleared; the summary carries
    # that context from now on.
    did_summarize = False
    if _estimate_tokens(unsummarized_messages) >= cfg.summary_interval:
        summary_prefix: list[dict] = []
        if sess.get("history_summary"):
            # Pass the existing summary as a system instruction so the model
            # knows to incorporate it, not ignore it.
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
        # Persist the new summary and record how many messages it now covers.
        await session_svc.save_summary(session_id, new_summary, prior_count)
        # Update the local sess dict so the assembly step below sees the new summary.
        sess = {**sess, "history_summary": new_summary, "last_summarized_count": prior_count}
        unsummarized_messages = []
        did_summarize = True

    # -------------------------------------------------------------------------
    # STEP 4 — Assemble conversation_history sent to the agent.
    #
    # The final list is built in layers (innermost = earliest in the list):
    #
    #   [0] facts block   (system) — personal preferences about this user
    #   [1] identity msg  (system) — who the user is (name, auth status, etc.)
    #   [2] summary msg   (system) — compressed history, if one exists
    #   [3..] unsummarized messages — recent raw turns not yet compressed
    #
    # The current user_message is NOT in this list; it is passed separately
    # to process_message() so the agent sees it as the "new" input.
    # -------------------------------------------------------------------------
    conversation_history: list[dict] = []

    # Layer 1: append compressed history as a system message (if it exists).
    if sess.get("history_summary"):
        conversation_history.append({
            "role": "system",
            "content": f"Summary of the conversation so far: {sess['history_summary']}",
        })

    # Layer 2: append the raw recent messages that aren't summarized yet.
    conversation_history.extend(unsummarized_messages)

    # Layer 3: prepend an identity system message (who the user is).
    identity_msg = await identity_svc.build_identity_message(
        authenticated_partner_id=principal.partner_id,
        session_partner_id=sess.get("partner_id"),
    )
    conversation_history = [{"role": "system", "content": identity_msg}] + conversation_history

    # Layer 4: prepend known facts about this user (only for identified users).
    # Facts are stored per-partner and injected as a system block so the agent
    # can personalise answers without re-discovering them via tool calls.
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
    #
    # process_message() drives the Claude tool-use loop. It returns the final
    # text reply and, if an OTP verification happened during this turn, the
    # newly confirmed partner_id (otherwise None).
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
    #
    # Skipped when verify_email_otp just ran in this same turn: that MCP tool
    # already touched the session row, and writing it again in the same
    # request causes a DB serialization conflict.
    # -------------------------------------------------------------------------
    if not verified_partner_id:
        try:
            await session_svc.touch_activity(session_id)
        except Exception as exc:
            _logger.warning("chat: touch_activity failed: %s", exc)

    return reply, did_summarize
