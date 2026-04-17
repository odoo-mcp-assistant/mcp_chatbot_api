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
from __future__ import annotations

import logging
from dataclasses import dataclass

from .odoo_client import get_client

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model_name: str


@dataclass(frozen=True)
class OdooConfig:
    mcp_server_url: str
    system_prompt: str
    max_tool_rounds: int
    summary_interval: int
    idle_timeout: int
    bot_name: str
    status: str
    llm: LLMConfig
    summary_llm: LLMConfig


_cached: OdooConfig | None = None


def _resolve_model(odoo, model_id_raw: str) -> str:
    """Look up mcp.llm.model → 'provider/name' (or 'name' if no provider)."""
    if not model_id_raw:
        return ""
    try:
        rec = odoo.env["mcp.llm.model"].browse(int(model_id_raw))
        if not rec.exists():
            return ""
        if rec.provider_id:
            return f"{rec.provider_id.name}/{rec.name}"
        return rec.name
    except Exception as exc:
        _logger.warning("resolve_model failed for id=%r: %s", model_id_raw, exc)
        return ""


def load_odoo_config() -> OdooConfig:
    """Synchronous read from Odoo. Called at startup (before the event
    loop gets busy) and from reload_odoo_config() via aodoo()."""
    global _cached
    odoo = get_client()
    param = odoo.env["ir.config_parameter"]

    def get(key: str, default: str = "") -> str:
        return param.get_param(key, default) or default

    main_api_key = get("mcp_chatbot.api_key")
    main_base_url = get("mcp_chatbot.base_url")
    main_model = _resolve_model(odoo, get("mcp_chatbot.llm_model_id"))

    summary_api_key = get("mcp_chatbot.summary_api_key") or main_api_key
    summary_base_url = get("mcp_chatbot.summary_base_url") or main_base_url
    summary_model = _resolve_model(odoo, get("mcp_chatbot.summary_model_id")) or main_model

    cfg = OdooConfig(
        mcp_server_url=get("mcp_chatbot.mcp_server_url"),
        system_prompt=get("mcp_chatbot.system_prompt"),
        max_tool_rounds=int(get("mcp_chatbot.max_tool_rounds", "5")),
        summary_interval=int(get("mcp_chatbot.summary_interval", "2000")),
        idle_timeout=int(get("mcp_chatbot.idle_timeout", "30")),
        bot_name=get("mcp_chatbot.bot_name", "AI Assistant"),
        status=get("mcp_chatbot.status", "online"),
        llm=LLMConfig(main_api_key, main_base_url, main_model),
        summary_llm=LLMConfig(summary_api_key, summary_base_url, summary_model),
    )
    _cached = cfg
    _logger.info(
        "odoo_config loaded: bot_name=%r, mcp_server=%r, model=%r, "
        "max_tool_rounds=%d, summary_interval=%d",
        cfg.bot_name, cfg.mcp_server_url, cfg.llm.model_name,
        cfg.max_tool_rounds, cfg.summary_interval,
    )
    return cfg


def get_odoo_config() -> OdooConfig:
    if _cached is None:
        raise RuntimeError(
            "OdooConfig not loaded — call load_odoo_config() at startup"
        )
    return _cached
