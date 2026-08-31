# Agent Applications

Put runnable, task-oriented NLIP agent systems in this directory. Each application
should integrate its agent(s) with `NLIPClient` and/or `NLIPServer` so configured
security extensions can observe and enforce NLIP traffic.

A typical simple application may look like:

```text
agent_applications/my_application/
├── agent.py          # Framework-native agent and task logic
├── nlip_agent.py     # NLIP client/server integration
├── config.toml       # Entities, endpoints, and enabled security extensions
└── README.md         # Scenario and run instructions
```

This structure is a suggestion, not a requirement. An application may contain one or
multiple agents, task environments, tools, or user clients. Security extension
implementations should remain in [`security_extensions`](../security_extensions/) and
be enabled by the application's configuration. See
[`nlip_examples`](../nlip_examples/) for reference agents and NLIP integrations.
