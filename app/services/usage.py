"""
Per-identity token usage accounting via odoorpc.

Thin wrapper over the Odoo model `mcp.chatbot.usage`, which stores one row
per (identity, UTC day). Two calls per chat turn:

  - `get_today_tokens(identity_key)` — read before the turn, compared by the
    router against the caller's daily budget.
  - `record_usage(identity_key, tokens, ...)` — written after the turn to add
    the tokens that turn actually billed (the summed `usage.total_tokens`
    from every LLM call in the turn).

Both delegate the get-or-create + increment to model methods on the Odoo
side so the read-modify-write stays inside one Odoo transaction.

Identity keys are produced by `app.ratelimit.identity_key_for` — "partner:<id>"
for logged-in users, "ip:<address>" for anonymous visitors.
"""

from ..odoo_client import get_client


async def get_today_tokens(identity_key: str) -> int:
    """Tokens already spent today by this identity (0 if none / on bad input)."""
    if not identity_key:
        return 0
    odoo = get_client()
    return odoo.env["mcp.chatbot.usage"].get_today_tokens(identity_key)


async def record_usage(
    identity_key: str,
    tokens: int,
    partner_id: int | None = None,
    ip_address: str | None = None,
) -> int:
    """Add `tokens` to today's row for this identity; returns the new total.

    partner_id / ip_address are stored for backend display only — the
    identity_key is the real dedup key.
    """
    if not identity_key:
        return 0
    odoo = get_client()
    return odoo.env["mcp.chatbot.usage"].record_usage(
        identity_key,
        int(tokens or 0),
        partner_id or False,
        ip_address or False,
    )
