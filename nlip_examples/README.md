# NLIP Examples

These examples show NLIP integration in two stages: first build an ordinary agent,
then add NLIP communication around it without rewriting its model logic.

## 1. Framework-native agents

`nlip_agnostic_agents/` contains two standalone chatbots:

- `ClaudeTestAgent`, built with the Claude Agent SDK;
- `LangChainTestAgent`, built with LangChain and an OpenAI-compatible model.

They manage their own model configuration and conversation history, but know nothing
about NLIP endpoints, messages, clients, or servers. Each file can also run directly as
an interactive chatbot.

```bash
python -m nlip_examples.nlip_agnostic_agents.claude_sdk_agent
python -m nlip_examples.nlip_agnostic_agents.langchain_agent
```

## 2. Add NLIP capabilities

`nlip_integration_examples/` imports those existing agents and composes each one with
the NLIP SDK:

```text
ClaudeTestAgent    + NLIPServer + NLIPClient -> ClaudeNLIPAgent
LangChainTestAgent + NLIPServer + NLIPClient -> LangChainNLIPAgent
```

The pieces have separate responsibilities:

- `NLIPServer` receives an NLIP HTTP request and serializes each session's complete
  server pipeline.
- The original agent produces the model response.
- `NLIPClient` sends messages to named entities and scopes outbound conversations to
  their parent session.
- `NLIPSession` owns the per-conversation lock and bounded NLIP request/response
  history. Each integration owns its framework-specific conversation objects.
- A small framework adapter exposes the same
  `send_nlip_message(destination, message)` tool to each model.

The integrated agents therefore work as both NLIP servers and NLIP clients. The
`UserChatClient` is simpler: it only needs `NLIPClient`, because it sends requests but
does not receive agent-to-agent requests. Conversation tokens let repeated messages
resume the same model history, while `/new` creates an independent conversation.

This separation keeps framework-specific model code independent from NLIP transport.
Both the shared server pipeline and NLIP client invoke the security extensions enabled
for that entity in `config.toml`.

See [the integration guide](nlip_integration_examples/README.md) for configuration and
the complete three-terminal demo.
