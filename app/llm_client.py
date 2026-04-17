"""
Cached AsyncOpenAI clients keyed by (api_key, base_url).

Instantiating AsyncOpenAI is not free — it sets up an httpx AsyncClient
underneath. Cache per-worker so we reuse connections across requests.
"""
from __future__ import annotations

from openai import AsyncOpenAI

_cache: dict[tuple[str, str], AsyncOpenAI] = {}


def get_async_openai(
    api_key: str,
    base_url: str,
    timeout: float = 118.0,
    max_retries: int = 2,
) -> AsyncOpenAI:
    key = (api_key, base_url)
    if key not in _cache:
        _cache[key] = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )
    return _cache[key]
