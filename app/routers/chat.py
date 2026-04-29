"""Chat + history + close + info endpoints. All require a JWT."""

import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ..auth import Principal, current_principal
from ..chat_pipeline import handle_chat, handle_chat_stream
from ..odoo_client import get_client
from ..odoo_config import get_odoo_config
from ..schemas import (
    CloseRequest,
    CloseResponse,
    HistoryMessage,
    HistoryResponse,
    InfoResponse,
    MessageRequest,
    MessageResponse,
)
from ..services import (
    fact_extractor as fact_extractor_svc,
    message as message_svc,
    session as session_svc,
)

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp_chatbot", tags=["chatbot"])


# ── Non-streaming message endpoint (backward compatible) ────────────────────

@router.post("/message", response_model=MessageResponse)
async def post_message(
    body: MessageRequest,
    principal: Principal = Depends(current_principal),
):
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

    reply, did_summarize = await handle_chat(principal, user_message)
    return MessageResponse(reply=reply, summarized=did_summarize)


# ── Streaming message endpoint (SSE) ────────────────────────────────────────

@router.post("/message/stream")
async def post_message_stream(
    body: MessageRequest,
    principal: Principal = Depends(current_principal),
):
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

    async def event_generator():
        try:
            async for event in handle_chat_stream(principal, user_message):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            _logger.exception("stream endpoint error: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal server error'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── History ─────────────────────────────────────────────────────────────────

@router.post("/history", response_model=HistoryResponse)
async def get_history(
    principal: Principal = Depends(current_principal),
):
    if principal.partner_id:
        sess = await session_svc.lookup_open_by_partner(principal.partner_id)
    elif principal.session_token:
        sess = await session_svc.lookup_by_token(principal.session_token)
    else:
        return HistoryResponse(status="not_found", messages=[])

    if not sess:
        return HistoryResponse(status="not_found", messages=[])
    if sess.get("state") == "closed":
        return HistoryResponse(status="closed", messages=[])

    msgs = await message_svc.list_by_session(sess["id"])
    return HistoryResponse(
        status="open",
        messages=[HistoryMessage(**m) for m in msgs],
    )


# ── Close ───────────────────────────────────────────────────────────────────

@router.post("/close", response_model=CloseResponse)
async def close_session(
    body: CloseRequest,
    background_tasks: BackgroundTasks,
    principal: Principal = Depends(current_principal),
):
    if principal.partner_id:
        sess = await session_svc.lookup_open_by_partner(principal.partner_id)
    elif principal.session_token:
        sess = await session_svc.lookup_by_token(principal.session_token)
    else:
        return CloseResponse()

    if not sess or sess.get("state") == "closed":
        return CloseResponse()

    session_id = sess["id"]
    if body.rating in ("bad", "neutral", "good"):
        try:
            await session_svc.save_rating(
                session_id=session_id,
                rating=body.rating,
                feedback=body.feedback or "",
                partner_id=sess.get("partner_id"),
            )
        except Exception as exc:
            _logger.warning("close: failed to save rating: %s", exc)

    await session_svc.close_session(session_id)

    session_partner_id = sess.get("partner_id")
    if session_partner_id:
        cfg = get_odoo_config()
        background_tasks.add_task(
            fact_extractor_svc.extract_and_save,
            session_id=session_id,
            partner_id=session_partner_id,
            llm=cfg.fact_llm,
        )

    return CloseResponse()


# ── Info ────────────────────────────────────────────────────────────────────

@router.post("/info", response_model=InfoResponse)
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