"""Chat + history + close + info endpoints. All require a JWT."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import Principal, current_principal
from ..chat_pipeline import handle_chat
from ..odoo_client import aodoo, get_client
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
from ..services import message as message_svc, session as session_svc

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp_chatbot", tags=["chatbot"])


@router.post("/message", response_model=MessageResponse)
async def post_message(
    body: MessageRequest,
    principal: Annotated[Principal, Depends(current_principal)],
):
    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message is required",
        )
    if not principal.is_authenticated and not principal.session_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_token is required for anonymous users",
        )

    reply, did_summarize = await handle_chat(principal, user_message)
    return MessageResponse(reply=reply, summarized=did_summarize)


@router.post("/history", response_model=HistoryResponse)
async def get_history(
    principal: Annotated[Principal, Depends(current_principal)],
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


@router.post("/close", response_model=CloseResponse)
async def close_session(
    body: CloseRequest,
    principal: Annotated[Principal, Depends(current_principal)],
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
    return CloseResponse()


@router.post("/info", response_model=InfoResponse)
async def get_info(
    principal: Annotated[Principal, Depends(current_principal)],
):
    cfg = get_odoo_config()

    first_name = ""
    if principal.partner_id:
        def _read_name():
            odoo = get_client()
            partner = odoo.env["res.partner"].browse(principal.partner_id)
            name = partner.read(["name"])[0].get("name") or ""
            return name.strip().split(" ")[0] if name else ""
        first_name = await aodoo(_read_name)

    return InfoResponse(
        bot_name=cfg.bot_name,
        status=cfg.status,
        is_authenticated=principal.is_authenticated,
        first_name=first_name,
    )
