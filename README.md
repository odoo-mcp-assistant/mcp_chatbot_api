# mcp_chatbot_api

FastAPI sidecar that handles the async LLM + MCP agentic loop for the Odoo `mcp_chatbot` addon.

## Why it exists

Odoo is a synchronous WSGI app. Every chat request used to tie up an Odoo worker for the full 10–30 s of the LLM + tool loop, so chatbot traffic competed with checkout and product pages for the same worker pool. Moving the long-running work into a dedicated async service lets the two scale independently, and keeps Odoo worker CPU time per chat message down to ~250–500 ms (just the JSON-RPC round-trips).

## Architecture

```
Browser widget ──► FastAPI (this service) ──► LLM + MCP server
                        │
                        └──odoorpc──► Odoo (sessions, messages, facts, settings)
```

- Widget calls FastAPI **directly** for chat, history, close, and info. Odoo is not in the hot path.
- Odoo only issues a short-lived JWT on page load, which identifies the user (`partner_id`) or the anonymous session (`session_token`).
- FastAPI reads/writes Odoo models via **odoorpc** (JSON-RPC) — no direct Postgres access, no duplicated schema. The calls are synchronous, so every one is wrapped in `asyncio.to_thread()`.
- Runtime settings (bot name, LLM keys, system prompt, MCP URL, thresholds) stay in `ir.config_parameter` and are snapshotted on startup via `load_odoo_config()`.

## Project layout

```
app/
  main.py           FastAPI app, lifespan (connect odoorpc → load config → init MCP), health endpoints
  config.py         .env / environment settings (pydantic-settings)
  odoo_client.py    odoorpc singleton + `aodoo(fn)` = asyncio.to_thread
  odoo_config.py    snapshot of mcp_chatbot.* ir_config_parameter values
  mcp_client.py     async MCP client (streamable HTTP transport), singleton
  llm_client.py     cached AsyncOpenAI clients keyed by (api_key, base_url)
  auth.py           JWT verification, `Principal` dependency
  schemas.py        pydantic request/response models
  chat_pipeline.py  per-request orchestrator: session → history+summary → facts → agent → persist
  agent.py          agentic loop — tool schemas, AUTH_REQUIRED_TOOLS gate, local remember_fact
  routers/chat.py   POST /mcp_chatbot/{message,history,close,info}
  services/         odoorpc wrappers: session, message, fact, identity, summary
```

## Setup

```bash
cd mcp_chatbot_api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with real Odoo credentials + JWT secret
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8020 --reload
```

On startup the service logs into Odoo, snapshots the chatbot config, and opens the MCP session. Any wiring issue fails loud — the process exits rather than starting half-broken.

## Configuration

All runtime values live in `.env` (see `.env.example`). The business config (bot name, LLM model, system prompt, MCP URL, summary interval, idle timeout) is read live from Odoo's `ir.config_parameter` by `load_odoo_config()` — edit it in **Odoo → Settings → Technical → MCP Chatbot**, then restart this service to pick up changes.

The `JWT_SECRET` in `.env` must match `mcp_chatbot.jwt_secret` in Odoo, otherwise every request 401s.

## Endpoints

Health (unauthenticated):

| Method | Path             | Purpose                                    |
|--------|------------------|--------------------------------------------|
| GET    | `/health`        | Liveness — always 200                      |
| GET    | `/health/odoo`   | odoorpc login still valid, returns user    |
| GET    | `/health/mcp`    | MCP session alive, lists advertised tools  |
| GET    | `/health/config` | Current config snapshot (no secrets)       |

Chat (all require `Authorization: Bearer <jwt>`):

| Method | Path                      | Purpose                                          |
|--------|---------------------------|--------------------------------------------------|
| POST   | `/mcp_chatbot/message`    | One user turn → assistant reply                  |
| POST   | `/mcp_chatbot/history`    | Full message list for the caller's open session  |
| POST   | `/mcp_chatbot/close`      | Close session, optionally save rating + feedback |
| POST   | `/mcp_chatbot/info`       | Bot name, status, auth state, first name         |

## JWT claims

Odoo mints the token; FastAPI verifies it. The token carries:

```json
{
  "sub":           "10" | "anon:<uuid>",
  "partner_id":    10,
  "session_token": null,
  "anonymous":     false,
  "iat":           1776449462,
  "exp":           1776453062,
  "aud":           "mcp-chatbot-api"
}
```

- Logged-in portal users get `partner_id` set, `session_token` null.
- Anonymous visitors get `partner_id` null, `session_token` = UUID from the browser's localStorage.
- OTP-verified anonymous sessions keep `session_token` but gain `session.partner_id` server-side — facts start being injected from that point.

## One-turn flow

1. Widget POSTs `/mcp_chatbot/message` with the bearer token.
2. `current_principal` decodes the JWT.
3. `chat_pipeline.handle_chat`:
   - resolves or creates the session (`session.get_or_create`),
   - writes the user message (`message.create`),
   - builds history: `[facts?] + [identity] + [summary?] + [recent messages]`,
   - if estimated tokens over `summary_interval`, asks the summary LLM to collapse history and persists the new summary,
   - runs `agent.process_message` — OpenAI chat completion + MCP tool loop, bounded by `max_tool_rounds`,
   - writes the assistant reply,
   - touches `last_activity` unless `verify_email_otp` already wrote to the session row this turn (avoids a serialization conflict).
4. Response: `{ "reply": "...", "summarized": true|false }`.

## Local development — generating a test JWT

Once the service is running, you can test without Odoo's token endpoint by minting a JWT from Python:

```bash
.venv/bin/python - <<'PY'
from dotenv import load_dotenv; load_dotenv()
import os, time
from jose import jwt
print(jwt.encode(
    {
        "sub": "10",
        "partner_id": 10,
        "session_token": None,
        "anonymous": False,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
        "aud": os.environ["JWT_AUDIENCE"],
    },
    os.environ["JWT_SECRET"],
    algorithm=os.environ["JWT_ALGORITHM"],
))
PY
```

Then:

```bash
TOKEN=<pasted>
curl -s -X POST http://localhost:8020/mcp_chatbot/info \
  -H "Authorization: Bearer $TOKEN" | jq
curl -s -X POST http://localhost:8020/mcp_chatbot/message \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"message":"hi, what do you know about me?"}' | jq
```

## Notes

- The MCP and odoorpc clients are **process-level singletons**. Run one uvicorn worker per process; if you scale to multiple uvicorn workers, each will open its own MCP session and odoorpc login — that is fine, but plan the MCP server capacity accordingly.
- `aodoo(lambda: odoo.env[...]...)` is the canonical way to touch Odoo from async code here. Never call odoorpc directly from a coroutine — it blocks the event loop.
- No raw SQL anywhere. All DB access goes through Odoo's ORM via odoorpc, matching the addon's existing style.
