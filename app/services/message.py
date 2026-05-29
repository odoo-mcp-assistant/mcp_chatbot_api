"""Message CRUD via odoorpc."""

from ..odoo_client import get_client


async def create(session_id: int, role: str, content: str) -> int:
    odoo = get_client()
    return odoo.env["mcp.chatbot.message"].create({
        "session_id": session_id,
        "role": role,
        "content": content,
    })
