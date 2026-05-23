"""Chat + history + close + info endpoints. All require a JWT."""

import logging

# APIRouter: groups related endpoints under a common prefix/tags — mounted on the app in main.py
# BackgroundTasks: schedules functions to run AFTER the HTTP response is sent to the client
# Depends: injects the result of another function (here: current_principal) into the route
# HTTPException: raised to return an HTTP error response (e.g. 400, 401)
# status: namespace of HTTP status code constants (status.HTTP_400_BAD_REQUEST = 400)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

# Principal: dataclass describing the authenticated caller (partner_id / session_token / anonymous)
# current_principal: FastAPI dependency that verifies the JWT and returns a Principal
from ..auth import Principal, current_principal

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
    HistoryMessage,
    HistoryResponse,
    InfoResponse,
    MessageRequest,
    MessageResponse,
)

# service layer — each module wraps odoorpc calls for one Odoo model
from ..services import (
    fact as fact_svc,
    message as message_svc,
    session as session_svc,
)

_logger = logging.getLogger(__name__)

# all endpoints in this router are served under /mcp_chatbot/...
# tags=["chatbot"] groups them together in the auto-generated /docs page
router = APIRouter(prefix="/mcp_chatbot", tags=["chatbot"])


# POST /mcp_chatbot/message — called by the widget when the user hits "Send"
# response_model=MessageResponse makes FastAPI validate and serialize the return value
@router.post("/message", response_model=MessageResponse)
async def post_message(
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
    if not principal.session_token and not principal.partner_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_token is required for anonymous users",
        )

    # delegate the real work to the pipeline — it runs the agent loop, persists messages,
    # and returns (reply_text, did_summarize) where did_summarize flags that history was compacted
    reply, did_summarize = await handle_chat(principal, user_message)
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
    msgs = await message_svc.list_by_session(sess["id"]) # list of dictionaries 
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

# fel fichier hedha aana 4 endpoints :
#   POST /message  → fih body (le user message) + header (JWT)
#   POST /close    → fih body (rating/feedback) + header (JWT)
#   GET  /history  → header khw (JWT) — read-only, no body
#   GET  /info     → header khw (JWT) — read-only, no body