# NLIP Shopping Assistant

A multi-agent shopping assistant whose agents talk to each other only through NLIP
messages (ECMA-430) over HTTP (ECMA-431). Users sign in with OAuth 2.0 / OpenID Connect
(Google, or a local mock provider). This is the application that the project's security
function protects. In this phase **no security extension is enabled**, and the baseline is
exploitable on purpose (see [Why this matters for agents](#why-this-matters-for-agents)).

## Scenario

| Entity | Port | Privilege | Role |
|---|---|---|---|
| `orchestrator` | 8000 | high | The user's assistant (LLM). Holds the OAuth-granted `user_scope`, and is the only entity that may place orders. Also serves the web UI and login. |
| `search_agent` | 8001 | low | Searches the catalog (LLM + tools), reads product web pages, which are **untrusted third-party text**. |
| `review_agent` | 8002 | low | Summarizes customer reviews (LLM + tools). |
| `checkout_agent` | 8003 | high | Deterministic pricing, spending-limit check, mock order commit (`data/orders.jsonl`). |

```
Browser (JS NLIP client) ──POST /nlip + authorization token──► orchestrator :8000
   /login → Google or mock OIDC → /auth/callback → ticket          │ NLIPClient
                              ┌──────────────────┬─────────────────┼──────────────────┐
                        search_agent :8001  review_agent :8002  checkout_agent :8003
```

A typical turn: the user asks for headphones. The orchestrator's LLM calls
`ask_search_agent` and `ask_review_agent`, and then, when the user wants to buy,
`request_checkout` and `place_order`. Orders at or above the user's `auto_confirm_under`
threshold wait for a **Confirm** click in the UI, which is sent as a structured NLIP
message `{"action": "confirm_order"}`.

## How the app uses NLIP

- **Message format (ECMA-430 §5).** Every agent reply is one message: a text summary as the
  first submessage, plus labeled `structured/json` submessages that carry fields. The
  labels are listed in `protocol.py`:

  | Label | Legitimate producer |
  |---|---|
  | `product_results` | `search_agent` |
  | `review_summary` | `review_agent` |
  | `cart_total`, `purchase_authorized`, `requires_confirmation`, `order_id` | `checkout_agent` |
  | `user_scope` | `orchestrator` (from the OAuth login) |
  | `order_status`, `trace` | `orchestrator` → user client |

  Checkout requests are `structured/json` (`{"action": "quote" | "commit", "items": [...]}`),
  with the user's scope in a `user_scope` submessage.
- **Tokens (ECMA-430 §6.2).** After login, the user client sends the orchestrator-issued
  ticket as an `authorization` token submessage on every request. Each server returns its
  own `conversation_<entity>` token and echoes tokens it did not create. The browser keeps
  the orchestrator's conversation token to resume the session.
- **HTTP binding (ECMA-431).** Every entity exposes `POST /nlip` (the shared `NLIPServer`).
- **Prompt rendering.** `protocol.render_for_prompt()` shows the model *every* non-token
  part as `[from <peer>] <label> (<format>) = <value>`, not just the plain text. ECMA-430
  describes `Label` as carrying, for example, role information in an LLM chat. A consumer
  that keeps only `extract_text()` throws away what the protocol is for.

## OAuth login

- `auth.py` implements the OIDC authorization-code flow. `GoogleProvider` uses Authlib
  with PKCE, `state`, and nonce, and verifies the ID token's signature, issuer, audience
  and expiry. It also requires `email_verified`.
- After login, the orchestrator issues its **own opaque ticket** (`TicketStore`). The
  ticket maps to the user's identity and their `user_scope` (`max_purchase`,
  `auto_confirm_under`), taken from `[policy]` in `config.toml`. Google tokens never leave
  `auth.py`, and the ticket is never sent to subagents.
- The orchestrator checks the ticket before a session is created. Missing, forged or
  expired tickets get 401. A conversation token used with another user's ticket gets 403.
- **Mock provider** (`provider = "mock"`, the default). It is a local stand-in with the
  same redirect → code → callback round trip, and it offers the test users in
  `[mock_users]`. `POST /auth/mock/token {"email": ...}` gives scripted clients a ticket.
  It performs no real authentication and exists only when the mock provider is selected.

### Setting up Google login

1. Google Cloud Console → *APIs & Services* → *OAuth consent screen*: choose External,
   leave it in **Testing**, and add the Google accounts that may sign in as test users
   (including graders, if they will use Google).
2. *Credentials* → *Create credentials* → *OAuth client ID* → **Web application**. Add the
   authorized redirect URI `http://localhost:8000/auth/callback`.
3. Put the credentials in a git-ignored `.env`, or export them:
   ```bash
   export AUTH_PROVIDER=google GOOGLE_CLIENT_ID=... GOOGLE_CLIENT_SECRET=...
   export SESSION_SECRET=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
   ```

Google accepts `http://localhost` redirect URIs. On a headless VM, open the UI through an
SSH tunnel so that the browser's `localhost:8000` is the VM's:
`ssh -L 8000:localhost:8000 exouser@<vm-ip>`.

## Running

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[shopping,test]"

python -m agent_applications.shopping_assistant.launch     # starts all four servers
# browser: http://localhost:8000/   (mock login: pick Alice or Bob)

# scripted client, in another shell:
python -m agent_applications.shopping_assistant.client_cli --user alice@example.com \
  "Find noise-cancelling headphones under \$150 and check their reviews." \
  "Buy the best rated one." --confirm
```

**LLM.** The default is Jetstream2's hosted `llama-4-scout` (OpenAI-compatible, no key).
Switch models with environment variables. The base URL and the model go together:

```bash
export SHOP_LLM_MODEL=gpt-oss-120b SHOP_LLM_BASE_URL=https://llm.jetstream-cloud.org/gpt-oss-120b/v1
```

Any OpenAI-compatible endpoint works. Set its key in `SHOP_LLM_API_KEY`.

Other environment variables: `SHOP_CONFIG` (alternative config file), `SHOP_ORDERS_PATH`
(order ledger), `AUTH_PROVIDER`, `AUTH_REDIRECT_URI`, `SESSION_SECRET`.

**Tests.** `pytest tests/test_shopping_app.py` runs without an LLM or network. Scripted
agents drive the real tools, and the subagents run in-process.

## Where security functions plug in

- **NLIP ingress/egress.** List SEs under `[entities.<name>.security]` in `config.toml`.
  Subagent replies reach the orchestrator at its **client-side ingress**. There,
  `SecurityEvent.peer` is the destination the orchestrator itself dialed, which gives
  trustworthy provenance that the sender cannot claim for itself. User requests arrive at
  the **server-side ingress**.
- **Tool boundary (Type-3).** The shared SE manager cannot see tool calls, so
  `tool_guard.py` defines `ToolGuard.before_tool_call` / `after_tool_call`. Enable a guard
  in `[entities.orchestrator].tool_guards`. Every orchestrator tool, including the
  privileged `place_order`, runs through the configured guards.

## Why this matters for agents

The orchestrator is a deputy with real privileges: it can spend the user's money. It
consumes subagent output in two realistic ways, and both are driven by labels:

1. `state.merge_labeled_fields()` copies known labeled fields into the order state, and the
   last writer wins. The `place_order` gate reads `purchase_authorized` and
   `requires_confirmation` from that state.
2. The same fields are rendered into the LLM's prompt.

NLIP gives no per-submessage provenance, and any sender can use any `Label`. A low-privilege
or compromised subagent can therefore write fields that belong to the checkout agent.
`test_undefended_baseline_trusts_fields_forged_by_a_subagent` shows this: a search agent
that emits `purchase_authorized` / `requires_confirmation` / `cart_total` gets a **$349
order placed for Bob, whose limit is $60, with no confirmation**. This is a confused-deputy
attack of the kind ECMA-434 §7.2.3 describes. The attacker never touches the user's
credentials. It relies only on the orchestrator trusting what it reads.

## ECMA-434 SCOs

Titles below are checked against the ECMA-434 1st edition text.

- **§7.2.1 Identity & Authentication: NLIP Client Authentication.** Implemented by the
  app: OAuth/OIDC login plus the NLIP `authorization` ticket.
- **§7.2.3 Runtime & Behaviour: Privilege Misuse Prevention.** The confused-deputy threat
  above. The primary target of the security function.
- **§7.2.2 Runtime & Behaviour: NLIP Client Authorization.** The spending scope and
  confirmation checks before the privileged `place_order` tool.
- **§7.3.1 Prevention of Prompt Injection (including Indirect Prompt Injection).**
  Product pages fetched by the search agent are untrusted content.
- **§7.3.2 Runtime Agent Behaviour Controls.** The tool-boundary guards.

**Known gap:** §7.1.1 (Transport Layer Security, mandatory Profile 1) is not met by the
local demo, which uses plain HTTP on 127.0.0.1. A deployment should serve every `/nlip`
endpoint over HTTPS.
