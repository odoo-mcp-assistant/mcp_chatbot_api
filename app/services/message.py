"""Message CRUD via odoorpc."""

from ..odoo_client import aodoo, get_client


async def create(session_id: int, role: str, content: str) -> int:
    def _sync() -> int:
        odoo = get_client()
        return odoo.env["mcp.chatbot.message"].create({
            "session_id": session_id,
            "role": role,
            "content": content,
        })
    return await aodoo(_sync)


async def list_by_session(session_id: int) -> list[dict[str, str]]:
    """Chronological list of [{role, content}, ...] for history replay."""
    def _sync() -> list[dict[str, str]]:
        odoo = get_client()
        Msg = odoo.env["mcp.chatbot.message"]
        ids = Msg.search(
            [("session_id", "=", session_id)],
            order="create_date asc, id asc",
        )
        if not ids:
            return []
        return [
            {"role": m["role"], "content": m["content"] or ""}
            for m in Msg.browse(ids).read(["role", "content"])
        ]
    return await aodoo(_sync)
