"""
Probe how an OpenAI-compatible provider (Groq/Kimi, etc.) streams chat
completions — with and without tool calls.

Run it BEFORE trusting the streaming agent: it prints every raw chunk the
provider sends, then reassembles them with the exact same logic as
app.agent._stream_completion, so you can verify the assumptions that logic
makes:

  1. tool_calls fragments are keyed by `index`
  2. `id` and `function.name` arrive once (first fragment), whole
  3. `function.arguments` arrives in string pieces to be concatenated
  4. which field carries reasoning deltas (reasoning_content vs reasoning)
  5. whether content tokens arrive in the same round as tool calls

Usage:
    export LLM_API_KEY=...                  # same key the chatbot uses
    export LLM_BASE_URL=...                 # e.g. https://api.groq.com/openai/v1
    export LLM_MODEL=...                    # e.g. moonshotai/kimi-k2-instruct
    python scripts/stream_probe.py          # both scenarios
    python scripts/stream_probe.py text     # text-only scenario
    python scripts/stream_probe.py tools    # tool-call scenario
"""

import asyncio
import os
import sys

from openai import AsyncOpenAI

# A fake tool mirroring the real get_orders schema, so the model is tempted
# to call it the same way it would in production.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_orders",
        "description": (
            "Get sale orders for the customer. partner_id is auto-injected, "
            "always pass null. Authentication required."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "description": "Page number starting from 1."},
                "partner_id": {"type": ["integer", "null"]},
            },
        },
    },
}]


def describe_chunk(i: int, chunk) -> str:
    """One compact line per chunk showing exactly what the provider sent."""
    if not chunk.choices:
        return f"[{i:03d}] (no choices — usage/keepalive frame)"
    choice = chunk.choices[0]
    delta = choice.delta
    parts = []
    if delta is not None:
        if getattr(delta, "role", None):
            parts.append(f"role={delta.role!r}")
        if getattr(delta, "content", None):
            parts.append(f"content={delta.content!r}")
        for field in ("reasoning_content", "reasoning"):
            if getattr(delta, field, None):
                parts.append(f"{field}={getattr(delta, field)!r}")
        for tc in getattr(delta, "tool_calls", None) or []:
            fn = tc.function
            parts.append(
                "tool_call(index=%r, id=%r, name=%r, arguments=%r)"
                % (tc.index, tc.id, fn.name if fn else None,
                   fn.arguments if fn else None)
            )
    if choice.finish_reason:
        parts.append(f"finish_reason={choice.finish_reason!r}")
    return f"[{i:03d}] " + ("  ".join(parts) if parts else "(empty delta)")


async def run_scenario(llm, model: str, title: str, messages: list, tools=None):
    print(f"\n{'=' * 70}\nSCENARIO: {title}\n{'=' * 70}")
    kwargs = {"model": model, "messages": messages, "stream": True}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    stream = await llm.chat.completions.create(**kwargs)

    # --- identical reassembly logic to app.agent._stream_completion ---
    content_parts, reasoning_parts = [], []
    tool_calls_by_index: dict[int, dict] = {}

    i = 0
    async for chunk in stream:
        print(describe_chunk(i, chunk))
        i += 1
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
        r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if r:
            reasoning_parts.append(r)
        for tc in getattr(delta, "tool_calls", None) or []:
            idx = tc.index if tc.index is not None else 0
            entry = tool_calls_by_index.setdefault(idx, {
                "id": "", "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if tc.id and not entry["id"]:
                entry["id"] = tc.id
            if tc.function:
                if tc.function.name and not entry["function"]["name"]:
                    entry["function"]["name"] = tc.function.name
                if tc.function.arguments:
                    entry["function"]["arguments"] += tc.function.arguments

    print(f"\n--- REASSEMBLED ({title}) ---")
    print(f"content   : {''.join(content_parts)!r}")
    print(f"reasoning : {''.join(reasoning_parts)[:300]!r}")
    for idx in sorted(tool_calls_by_index):
        print(f"tool_call[{idx}]: {tool_calls_by_index[idx]}")
    if not tool_calls_by_index:
        print("tool_calls: (none)")


async def main():
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    if not (api_key and model):
        sys.exit("Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL env vars first.")

    llm = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"

    if which in ("text", "both"):
        await run_scenario(
            llm, model, "text only (no tools offered)",
            [{"role": "user", "content": "In two short sentences, what is Odoo?"}],
        )

    if which in ("tools", "both"):
        await run_scenario(
            llm, model, "tool call (narration + tool expected)",
            [
                {"role": "system", "content": (
                    "You are a store assistant with tools. Before calling a "
                    "tool, briefly tell the user what you are about to do."
                )},
                {"role": "user", "content": "Show me my orders please."},
            ],
            tools=TOOLS,
        )


if __name__ == "__main__":
    asyncio.run(main())
