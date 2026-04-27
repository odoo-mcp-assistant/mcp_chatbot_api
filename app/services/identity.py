"""
Build the identity system message injected into every LLM call.

Three user types:
  1. Fully anonymous — not logged in, no OTP verification.
  2. OTP-verified contact — partner exists, no portal account.
  3. Portal login OR OTP-verified with an existing portal account.
"""

from ..odoo_client import get_client


async def build_identity_message(
    authenticated_partner_id: int | None,
    session_partner_id: int | None,
) -> str:
    # Type 3 — portal / internal login via Odoo session (token-verified on our side)
    if authenticated_partner_id:
        odoo = get_client()
        vals = odoo.env["res.partner"].browse(authenticated_partner_id).read(["name"])[0]
        return (
            f"Current authenticated user (portal account): "
            f"name='{vals.get('name') or ''}'."
        )

    # Type 2 or 3-edge — OTP-verified partner linked to the session
    if session_partner_id:
        odoo = get_client()
        vals = odoo.env["res.partner"].browse(session_partner_id).read(
            ["name", "email", "user_ids"]
        )[0]
        name = vals.get("name") or ""
        email = vals.get("email") or ""
        if vals.get("user_ids"):
            return (
                f"Current user: verified via email OTP and has a portal account. "
                f"name='{name}', email='{email}'. "
                f"They can use all authentication-required actions."
            )
        return (
            f"Current user: verified via email OTP (contact only, no portal account). "
            f"name='{name}', email='{email}'. "
            f"They can use authentication-required actions."
        )

    # Type 1 — fully anonymous
    return "Current user: not logged in (anonymous visitor)."
