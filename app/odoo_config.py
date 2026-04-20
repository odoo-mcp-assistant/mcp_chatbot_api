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
    mcp_chatbot.summary_api_key      (falls back to main)
    mcp_chatbot.summary_base_url     (falls back to main)
    mcp_chatbot.summary_model_id     (falls back to main)
"""


# logging: used to print info/warning messages from this module
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
    api_key: str    # the API key to authenticate with the LLM provider (e.g. NVIDIA, OpenAI)
    base_url: str   # the base URL of the LLM API (e.g. https://integrate.api.nvidia.com/v1)
    model_name: str # the exact model identifier to use (e.g. "zhipuai/glm-4-9b")


# OdooConfig holds ALL chatbot settings loaded from Odoo — one snapshot per startup
# frozen=True means once loaded, nothing can accidentally mutate these values
@dataclass(frozen=True)
class OdooConfig:
    mcp_server_url: str    # URL of the MCP server that exposes Odoo tools to the AI
    system_prompt: str     # the initial instruction given to the LLM at the start of every conversation
    max_tool_rounds: int   # maximum number of times the AI can call tools in one message turn before being forced to reply
    summary_interval: int  # estimated token count after which conversation history gets compressed into a summary
    idle_timeout: int      # minutes of user inactivity before a session is automatically closed
    bot_name: str          # display name of the chatbot shown in the widget
    status: str            # "online" or "offline" — controls whether the widget accepts new messages
    llm: LLMConfig         # config for the main LLM (used for the agent loop and tool calls)
    summary_llm: LLMConfig # config for the summary LLM (used to compress long conversations — can be a cheaper/faster model)


# module-level variable that stores the loaded config in memory
# None means config hasn't been loaded yet — set once at startup by load_odoo_config()
_cached: OdooConfig | None = None


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
            # model has a provider (e.g. provider="zhipuai", name="glm-4-9b")
            # combine them as "zhipuai/glm-4-9b" — the format NVIDIA and most APIs expect
            return f"{rec.provider_id.name}/{rec.name}"

        # model has no provider — return just the name as-is
        return rec.name

    except Exception as exc:
        # log a warning instead of crashing — the agent loop will fail later with a clearer error
        _logger.warning("resolve_model failed for id=%r: %s", model_id_raw, exc)
        return ""


def load_odoo_config() -> OdooConfig:
    """Synchronous read from Odoo. Called at startup (before the event
    loop gets busy) and from reload_odoo_config() via aodoo()."""

    global _cached          # we will write to the module-level _cached variable
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
    # --- Build the immutable config snapshot ---
    cfg = OdooConfig(
        mcp_server_url  = param.get_param("mcp_chatbot.mcp_server_url") or "",               
        system_prompt   = param.get_param("mcp_chatbot.system_prompt") or "",                
        max_tool_rounds = int(param.get_param("mcp_chatbot.max_tool_rounds") or "5"),        
        summary_interval= int(param.get_param("mcp_chatbot.summary_interval") or "2000"),   
        idle_timeout    = int(param.get_param("mcp_chatbot.idle_timeout") or "30"),          
        bot_name        = param.get_param("mcp_chatbot.bot_name") or "AI Assistant",        
        status          = param.get_param("mcp_chatbot.status") or "online",                
        llm             = LLMConfig(main_api_key, main_base_url, main_model),               
        summary_llm     = LLMConfig(summary_api_key, summary_base_url, summary_model),   
    )

    _cached = cfg   # store in memory so get_odoo_config() can return it without re-reading Odoo

    # log a confirmation so we can verify the correct settings were loaded at startup
    _logger.info(
        "odoo_config loaded: bot_name=%r, mcp_server=%r, model=%r, "
        "max_tool_rounds=%d, summary_interval=%d",
        cfg.bot_name, cfg.mcp_server_url, cfg.llm.model_name,
        cfg.max_tool_rounds, cfg.summary_interval,
    )
    return cfg


def get_odoo_config() -> OdooConfig:
    # crash loudly if called before load_odoo_config() ran at startup
    # this should never happen in normal operation — it means the lifespan hook didn't run
    if _cached is None:
        raise RuntimeError(
            "OdooConfig not loaded — call load_odoo_config() at startup"
        )
    return _cached  # return the in-memory snapshot — no Odoo call needed
