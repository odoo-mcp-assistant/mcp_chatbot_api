"""
User-fact CRUD via odoorpc.

Facts are durable per-partner records written by the `remember_fact`
local tool inside the agentic loop. They're read whole on every chat
request and injected as a single system message — no retrieval/ranking.
"""

import logging

from ..odoo_client import get_client

_logger = logging.getLogger(__name__)


async def list_for_partner(partner_id: int) -> list[dict]:
    """Newest first. Returns [{id, fact_text, category}, ...]."""
    odoo = get_client()
    Fact = odoo.env["mcp.chatbot.user.fact"]
    ids = Fact.search([("partner_id", "=", partner_id)], order="create_date desc")
    if not ids:
        return []
    return Fact.browse(ids).read(["id", "fact_text", "category"])


async def save(partner_id: int, fact_text: str, category: str = "general") -> int:
    odoo = get_client()
    return odoo.env["mcp.chatbot.user.fact"].create({
        "partner_id": partner_id,
        "fact_text": fact_text,
        "category": category,
    })
