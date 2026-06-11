"""Chat + history + close + info endpoints. All require a JWT."""

import logging

# APIRouter: groups related endpoints under a common prefix/tags — mounted on the app in main.py
# BackgroundTasks: schedules functions to run AFTER the HTTP response is sent to the client
# Depends: injects the result of another function (here: current_principal) into the route
# HTTPException: raised to return an HTTP error response (e.g. 400, 401)
# Request: the raw request object — slowapi needs it to read the caller's rate-limit key
# status: namespace of HTTP status code constants (status.HTTP_400_BAD_REQUEST = 400)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

# Principal: dataclass describing the authenticated caller (partner_id / session_token / anonymous)
# current_principal: FastAPI dependency that verifies the JWT and returns a Principal
from ..auth import Principal, current_principal

# get_settings: reads the .env config; used here for the message-length cap and
# rate-limit strings so both stay configurable without code changes
from ..config import get_settings

# limiter: shared slowapi throttler — its .limit() decorator caps how fast a
# single caller may POST /message
# identity_key_for: the per-caller key ("partner:<id>" / "ip:<addr>") shared by
# the rate limiter and the daily token budget (see ratelimit.py)
from ..ratelimit import limiter, identity_key_for

# handle_chat: the orchestrator that runs the agent loop, saves messages, and triggers summaries
from ..chat_pipeline import handle_chat

# get_client: returns the shared odoorpc connection (opened once at startup)
from ..odoo_client import get_client

# get_odoo_config: returns the cached Odoo settings snapshot (bot name, status, LLM config, etc.)
from ..odoo_config import get_odoo_config

# Pydantic request/response models — FastAPI uses them for validation + OpenAPI docs
from ..schemas import (
    CloseRequest,
    CloseResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationSummary,
    HistoryMessage,
    HistoryResponse,
    InfoResponse,
    MessageRequest,
    MessageResponse,
)

# service layer — each module wraps odoorpc calls for one Odoo model
from ..services import (
    fact as fact_svc,
    session as session_svc,
    usage as usage_svc,
)

_logger = logging.getLogger(__name__)

# all endpoints in this router are served under /mcp_chatbot/...
# tags=["chatbot"] groups them together in the auto-generated /docs page
router = APIRouter(prefix="/mcp_chatbot", tags=["chatbot"])


# POST /mcp_chatbot/message — called by the widget when the user hits "Send"
# response_model=MessageResponse makes FastAPI validate and serialize the return value
#
# The two @limiter.limit decorators throttle this endpoint per caller (keyed by
# partner_id when logged in, else client IP — see ratelimit.py). A request that
# trips either ceiling is rejected with HTTP 429 before any LLM work happens.
# The limits are read from .env via lambdas so they can be tuned without a code
# change. Both decorators require the `request: Request` parameter below.
@router.post("/message", response_model=MessageResponse)
@limiter.limit(lambda: get_settings().rate_limit_message_burst)  # short-window: stop rapid-fire
@limiter.limit(lambda: get_settings().rate_limit_message)        # sustained: cap total volume
async def post_message(
    request: Request,       # required by slowapi to identify the caller; not used directly
    body: MessageRequest,   # request body parsed and validated against MessageRequest
    # Depends(current_principal) tells FastAPI to call current_principal first, verify the JWT,
    # and pass the resulting Principal into this function as `principal`
    principal: Principal = Depends(current_principal),
):
    # trim whitespace — if the message is empty after stripping, reject it
    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message is required",
        )
    # Length cap — every character is eventually billed by the LLM, so reject
    # oversized messages (e.g. someone pasting a whole document) before the
    # agent loop spends any tokens. Limit is configurable via .env.
    max_length = get_settings().max_message_length
    if len(user_message) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"message is too long (max {max_length} characters)",
        )
    if not principal.session_token and not principal.partner_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_token is required for anonymous users",
        )

    cfg = get_odoo_config()

    # Availability gate — the real enforcement of mcp_chatbot.status. The
    # widget already refuses to send when the status isn't 'online', but that
    # is cosmetic: anyone holding a valid JWT can call this endpoint directly.
    # Default-deny (anything other than 'online' blocks) so a missing or
    # unexpected status value never lets traffic through. Checked BEFORE the
    # pipeline so an offline chatbot creates no session, persists nothing, and
    # spends no LLM tokens.
    if cfg.status != "online":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot is currently offline.",
        )

    # Per-identity daily token budget. Same key as the rate limiter:
    # "partner:<id>" for logged-in users, "ip:<addr>" for anonymous visitors —
    # so one heavy user is throttled in isolation and can never exhaust a budget
    # shared by everyone else. A budget of 0 disables that tier.
    identity_key = identity_key_for(principal, request)
    budget = (
        cfg.daily_token_budget_authenticated
        if principal.partner_id
        else cfg.daily_token_budget_anonymous
    )
    if budget > 0:
        try:
            used_today = await usage_svc.get_today_tokens(identity_key)
        except Exception as exc:
            # Fail open: a usage-table glitch must not take chat down.
            _logger.warning("message: usage read failed for %s: %s", identity_key, exc)
            used_today = 0

        # Anonymous visitors can lift their ceiling for the current session:
        #   - +verified_anonymous_bonus once the session is OTP-verified
        #   - +otp_pending_grace while verification is in progress (so an
        #     over-budget visitor can still finish the OTP + order flow)
        # The verified bonus supersedes the grace (not stacked). We only pay for
        # the extra session lookup once the caller has actually hit the base
        # budget — under it, the bonus can't change the allow/deny outcome.
        effective_budget = budget
        if (
            used_today >= budget
            and not principal.partner_id
            and principal.session_token
        ):
            try:
                sess = await session_svc.lookup_by_token(principal.session_token)
            except Exception as exc:
                _logger.warning("message: session lookup failed for budget grace: %s", exc)
                sess = None
            if sess and sess.get("state") == "open":
                if sess.get("partner_id"):
                    effective_budget += cfg.verified_anonymous_bonus
                elif sess.get("otp_pending"):
                    effective_budget += cfg.otp_pending_grace

        if used_today >= effective_budget:
            # Structured detail so the widget can tell this apart from a
            # rate-limit 429 (which means "slow down", not "done for today").
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "daily_budget_exceeded",
                    "message": "You've reached today's usage limit. Please try again tomorrow.",
                },
            )

    # delegate the real work to the pipeline — it runs the agent loop, persists
    # messages, and returns (reply_text, did_summarize, tokens_used) where
    # did_summarize flags that history was compacted and tokens_used is this
    # turn's total provider-reported token cost.
    reply, did_summarize, tokens_used = await handle_chat(principal, user_message)

    # Charge this turn's tokens to the caller. Best-effort: an accounting write
    # must never fail the user's reply. The check above runs on the PRE-turn
    # total, so a caller can overshoot by at most one turn — acceptable.
    try:
        await usage_svc.record_usage(
            identity_key,
            tokens_used,
            partner_id=principal.partner_id,
            ip_address=identity_key[3:] if identity_key.startswith("ip:") else None,
        )
    except Exception as exc:
        _logger.warning("message: usage record failed for %s: %s", identity_key, exc)

    return MessageResponse(reply=reply, summarized=did_summarize)


# GET /mcp_chatbot/history — called by the widget on load to restore previous messages in the chat window
@router.get("/history", response_model=HistoryResponse)
async def get_history(
    principal: Principal = Depends(current_principal),
):
    # look up the session differently depending on who the caller is
    if principal.partner_id:
        # logged-in user — find their most recent OPEN session
        sess = await session_svc.lookup_open_by_partner(principal.partner_id)
    elif principal.session_token:
        # anonymous user — find the session attached to their token (open or closed)
        sess = await session_svc.lookup_by_token(principal.session_token)
    else:
        # no identity at all → no history to return
        return HistoryResponse(status="not_found", messages=[])

    if not sess:
        return HistoryResponse(status="not_found", messages=[])
    # if the session exists but has been closed, return empty messages with status="closed"
    # so the frontend can decide whether to start a new one
    if sess.get("state") == "closed":
        return HistoryResponse(status="closed", messages=[])

    # session is open → fetch its messages in chronological order and return them
    msgs = await session_svc.get_conversation_history(sess["id"]) # list of dictionaries 
    return HistoryResponse(
        status="open",
        # unpack each {"role": ..., "content": ...} dict into a HistoryMessage pydantic model using the ** instaed of passing them seperatly role:.... content:......
        messages=[HistoryMessage(**m) for m in msgs],
    )


# POST /mcp_chatbot/close — called by the widget when the user submits a rating or explicitly ends the conversation
@router.post("/close", response_model=CloseResponse)
async def close_session(
    body: CloseRequest,
    background_tasks: BackgroundTasks,
    principal: Principal = Depends(current_principal),
):
    # same lookup logic as /history — find the caller's open session
    if principal.partner_id:
        sess = await session_svc.lookup_open_by_partner(principal.partner_id)
    elif principal.session_token:
        sess = await session_svc.lookup_by_token(principal.session_token)
    else:
        # nothing to close → respond success (idempotent)
        return CloseResponse()

    # nothing to close if no session exists or it's already closed
    if not sess or sess.get("state") == "closed":
        return CloseResponse()

    session_id = sess["id"]
    # if the caller included a valid rating, persist it as a separate rating record
    # (ratings are a distinct Odoo model linked to the session)
    if body.rating in ("bad", "neutral", "good"):
        try:
            await session_svc.save_rating(
                session_id=session_id,
                rating=body.rating,
                feedback=body.feedback or "",
                partner_id=sess.get("partner_id"),
            )
        except Exception as exc:
            # rating save is non-critical — log and carry on so we still close the session
            _logger.warning("close: failed to save rating: %s", exc)

    # mark the Odoo session as closed
    await session_svc.close_session(session_id)

    # Fact extraction runs an LLM call (slow) and the user doesn't need its result,
    # so we hand it to FastAPI's BackgroundTasks: add_task() only queues the call —
    # it executes AFTER CloseResponse has been sent, so the widget unblocks immediately.
    # Skipped for anonymous sessions — without a partner_id there's nowhere to save facts.
    session_partner_id = sess.get("partner_id")
    if session_partner_id:
        cfg = get_odoo_config()
        background_tasks.add_task(
            fact_svc.extract_and_save,
            session_id=session_id,
            partner_id=session_partner_id,
            llm=cfg.fact_llm,
        )

    return CloseResponse()


# GET /mcp_chatbot/info — called by the widget on load to get the bot name and greet the user by first name
@router.get("/info", response_model=InfoResponse)
async def get_info(
    principal: Principal = Depends(current_principal),
):
    cfg = get_odoo_config()

    first_name = ""
    if principal.partner_id:
        odoo = get_client()
        partner = odoo.env["res.partner"].browse(principal.partner_id)
        name = partner.read(["name"])[0].get("name") or ""
        first_name = name.strip().split(" ")[0] if name else ""

    return InfoResponse(
        bot_name=cfg.bot_name,
        status=cfg.status,
        first_name=first_name,
        is_authenticated=bool(principal.partner_id),
    )

# GET /mcp_chatbot/conversations — sidebar list of the caller's past conversations.
# Authenticated users only: anonymous visitors have no durable, browsable history
# (their identity is a throwaway sessionStorage token), so we reject them with 403.
@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    principal: Principal = Depends(current_principal),
):
    if not principal.partner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Conversation history is available to logged-in users only",
        )
    convos = await session_svc.list_by_partner(principal.partner_id)
    return ConversationListResponse(
        conversations=[ConversationSummary(**c) for c in convos],
    )


# GET /mcp_chatbot/conversations/{session_id} — read one past conversation.
# The service enforces ownership (the session must belong to this partner);
# a missing or someone-else's session both surface as 404 so ids can't be probed.
@router.get("/conversations/{session_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    session_id: int,
    principal: Principal = Depends(current_principal),
):
    if not principal.partner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Conversation history is available to logged-in users only",
        )
    result = await session_svc.get_owned_history(session_id, principal.partner_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    state, msgs = result
    return ConversationDetailResponse(
        id=session_id,
        status="ok",
        state=state,
        messages=[HistoryMessage(**m) for m in msgs],
    )


# fel fichier hedha aana 4 endpoints :
#   POST /message  → fih body (le user message) + header (JWT)
#   POST /close    → fih body (rating/feedback) + header (JWT)
#   GET  /history  → header khw (JWT) — read-only, no body
#   GET  /info     → header khw (JWT) — read-only, no body