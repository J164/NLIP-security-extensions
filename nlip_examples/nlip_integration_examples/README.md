# NLIP Integration Example

This demo runs three entities:

- a user chat client;
- a Claude agent that is both an NLIP server and client;
- a LangChain agent that is both an NLIP server and client.

Both agents expose one model tool:

```text
send_nlip_message(destination, message)
```

The destination names and endpoints come from `config.toml`. The messages use the
ECMA-430 NLIP format over the ECMA-431 HTTP binding. Each conversation has its own
opaque conversation tokens, bounded NLIP history, model history, and outbound agent
sessions.

## 1. Install

From the `NLIP-security-extensions` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[agents]'
```

Claude uses your local Claude Code subscription. If necessary, run `claude` once and
sign in before starting the demo.

LangChain requires an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="http://your-litellm-server:4000/v1"  # omit for OpenAI
```

## 2. Check the configuration

The default `config.toml` uses:

```text
Claude:    http://127.0.0.1:8011/nlip
LangChain: http://127.0.0.1:8012/nlip
```

Change `host`, `port`, or `endpoint` there if either port is already in use. Adding
another `[entities.<name>]` entry automatically adds it to the NLIP client's available
destinations.

The same file enables `AuditSecurityExtension` at both `ingress` and `egress` for each
agent. It prints the observed message, direction, transport side, session, and current-
turn context in the corresponding agent terminal, then always allows the message. To
run without audit output, change both security lists for that entity to `[]`.

## 3. Start the demo

Open three terminals, activate the same virtual environment in each, and run the
following commands from the repository root.

Terminal 1 — LangChain agent:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="http://your-litellm-server:4000/v1"  # omit for OpenAI
python -m nlip_examples.nlip_integration_examples.langchain_nlip_agent
```

Terminal 2 — Claude agent:

```bash
python -m nlip_examples.nlip_integration_examples.claude_sdk_nlip_agent
```

Terminal 3 — user client:

```bash
python -m nlip_examples.nlip_integration_examples.user_chat_client
```

## 4. Try it

Directly talk to either agent:

```text
/agents
/use langchain
Who are you?
/use claude
Who are you?
```

Messages continue the current conversation. Run `/new` to start a fresh session with
the selected agent without changing sessions for the other agents.

Then ask Claude to contact LangChain:

```text
Ask the LangChain agent "Who are you?" and tell me its response.
```

The message path is:

```text
user -> Claude NLIP server -> Claude tool -> LangChain NLIP server -> Claude -> user
```

Use `/exit` to stop the user client and `Ctrl-C` to stop both agent servers. If an NLIP
request reports `Connection refused`, check that the destination server is running and
that its endpoint matches `config.toml`. Session state is intentionally process-local for
this demo, so restarting an agent starts fresh conversations.
