# Framework-Native Agents

This directory contains two minimal, multi-turn chatbots:

- `ClaudeTestAgent` uses the Claude Agent SDK and the local Claude Code login.
- `LangChainTestAgent` uses LangChain with OpenAI or an OpenAI-compatible endpoint.

Install the agent dependencies from the repository root:

```bash
pip install -e '.[agents]'
```

Run the Claude chatbot after signing in through the local `claude` command:

```bash
python -m nlip_examples.nlip_agnostic_agents.claude_sdk_agent
```

Run the LangChain chatbot after setting `OPENAI_API_KEY`; optionally set
`OPENAI_BASE_URL` for a compatible server:

```bash
python -m nlip_examples.nlip_agnostic_agents.langchain_agent
```

These agents do not depend on NLIP. They contain only framework-native model and
conversation logic; the sibling
[`nlip_integration_examples`](../nlip_integration_examples/) shows how to add NLIP
client and server capabilities around them.
