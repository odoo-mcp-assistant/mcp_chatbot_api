"""
Per-chatbot settings read from Odoo's `ir.config_parameter`.

The Odoo admin edits these via Settings UI. This module snapshots them
into a plain dataclass at startup and on demand. Reloaded when the admin
saves settings (future: via a `/reload_config` endpoint the addon hits).

Keys read
---------
    mcp_chatbot.mcp_server_url
    mcp_chatbot.api_key              → main LLM key
    mcp_chatbot.base_url             → main LLM base URL
    mcp_chatbot.llm_model_id         → FK to mcp.llm.model
    mcp_chatbot.system_prompt
    mcp_chatbot.max_tool_rounds      (default 5)
    mcp_chatbot.summary_interval     (default 2000 estimated tokens)
    mcp_chatbot.idle_timeout         (default 30 minutes)
    mcp_chatbot.bot_name             (default "AI Assistant")
    mcp_chatbot.status               (default "online")
    mcp_chatbot.daily_token_budget_authenticated  (default 500000; 0 disables)
    mcp_chatbot.daily_token_budget_anonymous      (default 300000; 0 disables)
    mcp_chatbot.verified_anonymous_bonus          (default 200000)
    mcp_chatbot.otp_pending_grace                 (default 30000)
    mcp_chatbot.summary_api_key      (falls back to main)
    mcp_chatbot.summary_base_url     (falls back to main)
    mcp_chatbot.summary_model_id     (falls back to main)
    mcp_chatbot.fact_api_key         (falls back to main)
    mcp_chatbot.fact_base_url        (falls back to main)
    mcp_chatbot.fact_model_id        (falls back to main)
"""


import logging

# dataclass: decorator that auto-generates __init__, __repr__, etc. from class fields
from dataclasses import dataclass

# get_client: returns the already-connected odoorpc client to read from Odoo
from .odoo_client import get_client

# create a logger for this file — messages will appear as "app.odoo_config" in logs
_logger = logging.getLogger(__name__)


# @dataclass(frozen=True): creates an immutable data container — fields cannot be changed after creation it also creates the __init__ 
# LLMConfig holds the 3 things needed to call any LLM API
@dataclass(frozen=True)
class LLMConfig:
    api_key: str    
    base_url: str   
    model_name: str # the exact model identifier to use (e.g. "zhipuai/glm-4-9b")


# OdooConfig holds ALL chatbot settings loaded from Odoo — one snapshot per startup
# frozen=True means once loaded, nothing can accidentally mutate these values
@dataclass(frozen=True)
class OdooConfig:
    mcp_server_url: str    
    system_prompt: str     
    max_tool_rounds: int   
    summary_interval: int
    idle_timeout: int
    bot_name: str
    status: str
    # Per-identity daily token budgets (0 = that budget is disabled).
    daily_token_budget_authenticated: int
    daily_token_budget_anonymous: int
    # Extra daily allowance for an anonymous session once it is OTP-verified.
    verified_anonymous_bonus: int
    # Grace granted while an anonymous session is mid-verification (otp_pending),
    # so a checkout in progress isn't cut off. Superseded by the verified bonus.
    otp_pending_grace: int
    llm: LLMConfig
    summary_llm: LLMConfig
    fact_llm: LLMConfig


# module-level variable that stores the loaded config in memory
# None means config hasn't been loaded yet — set once at startup by load_odoo_config()
_cached: OdooConfig | None   = None


def _resolve_model(odoo, model_id_raw: str) -> str:
    """Look up mcp.llm.model → 'provider/name' (or 'name' if no provider)."""

    if not model_id_raw:   # if no model ID was configured in Odoo, return empty string
        return ""
    try:
        # browse the mcp.llm.model record in Odoo using the ID stored in ir.config_parameter
        rec = odoo.env["mcp.llm.model"].browse(int(model_id_raw))

        if not rec.exists():   # if the record was deleted from Odoo, return empty string
            return ""

        if rec.provider_id:
            return f"{rec.provider_id.name}/{rec.name}"

        # model has no provider — return just the name as-is
        return rec.name

    except Exception as exc:
        _logger.warning("resolve_model failed for id=%r: %s", model_id_raw, exc)
        return ""


def load_odoo_config() -> OdooConfig:
    """Synchronous read from Odoo. Called at startup (before the event
    loop gets busy) and from the /reload_config endpoint."""

    global _cached          
    odoo = get_client()     # get the connected odoorpc client
    param = odoo.env["ir.config_parameter"]   # Odoo's key-value settings table

    # --- Read main LLM settings ---
    main_api_key  = param.get_param("mcp_chatbot.api_key") or ""                            
    main_base_url = param.get_param("mcp_chatbot.base_url") or ""                           
    main_model    = _resolve_model(odoo, param.get_param("mcp_chatbot.llm_model_id") or "") 

    # --- Read summary LLM settings (fall back to main LLM if not configured separately) ---
    summary_api_key  = param.get_param("mcp_chatbot.summary_api_key")  or main_api_key
    summary_base_url = param.get_param("mcp_chatbot.summary_base_url") or main_base_url
    summary_model    = _resolve_model(odoo, param.get_param("mcp_chatbot.summary_model_id") or "") or main_model

    # --- Read fact-extraction LLM settings (fall back to main LLM if not configured separately) ---
    fact_api_key  = param.get_param("mcp_chatbot.fact_api_key")  or main_api_key
    fact_base_url = param.get_param("mcp_chatbot.fact_base_url") or main_base_url
    fact_model    = _resolve_model(odoo, param.get_param("mcp_chatbot.fact_model_id") or "") or main_model

    # --- Build the immutable config snapshot ---
    cfg = OdooConfig(
        mcp_server_url  = param.get_param("mcp_chatbot.mcp_server_url") or "",
        system_prompt   = param.get_param("mcp_chatbot.system_prompt") or "",
        max_tool_rounds = int(param.get_param("mcp_chatbot.max_tool_rounds") or "5"),
        summary_interval= int(param.get_param("mcp_chatbot.summary_interval") or "2000"),
        idle_timeout    = int(param.get_param("mcp_chatbot.idle_timeout") or "30"),
        bot_name        = param.get_param("mcp_chatbot.bot_name") or "AI Assistant",
        status          = param.get_param("mcp_chatbot.status") or "online",
        daily_token_budget_authenticated = int(param.get_param("mcp_chatbot.daily_token_budget_authenticated") or "500000"),
        daily_token_budget_anonymous     = int(param.get_param("mcp_chatbot.daily_token_budget_anonymous") or "300000"),
        verified_anonymous_bonus         = int(param.get_param("mcp_chatbot.verified_anonymous_bonus") or "200000"),
        otp_pending_grace                = int(param.get_param("mcp_chatbot.otp_pending_grace") or "30000"),
        llm             = LLMConfig(main_api_key, main_base_url, main_model),
        summary_llm     = LLMConfig(summary_api_key, summary_base_url, summary_model),
        fact_llm        = LLMConfig(fact_api_key, fact_base_url, fact_model),
    )

    _cached = cfg   # store in memory so get_odoo_config() can return it without re-reading Odoo

    # log a confirmation so we can verify the correct settings were loaded at startup
    _logger.info(
        "odoo_config loaded: bot_name=%r, mcp_server=%r, model=%r, "
        "max_tool_rounds=%d, summary_interval=%d, "
        "daily_budget(auth=%d, anon=%d)",
        cfg.bot_name, cfg.mcp_server_url, cfg.llm.model_name,
        cfg.max_tool_rounds, cfg.summary_interval,
        cfg.daily_token_budget_authenticated, cfg.daily_token_budget_anonymous,
    )
    return cfg

# we don't use @lru_cache because we need to refresh values if admin changes them 
def get_odoo_config() -> OdooConfig:
    # crash loudly if called before load_odoo_config() ran at startup
    # this should never happen in normal operation — it means the lifespan hook didn't run
    if _cached is None:
        raise RuntimeError(
            "OdooConfig not loaded — call load_odoo_config() at startup"
        )
    return _cached  # return the in-memory snapshot — no Odoo call needed
