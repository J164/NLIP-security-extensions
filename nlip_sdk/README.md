# NLIP SDK

This package provides the shared NLIP message, transport, session, configuration, and
security-extension infrastructure used by the examples and agent applications.

| File or folder | Key APIs | Purpose |
| --- | --- | --- |
| `__init__.py` | Public SDK exports | Provides the commonly used imports from `nlip_sdk`. |
| `config.py` | `load_config()` | Loads the shared TOML configuration and validates entity endpoints. |
| `nlip.py` | `NLIPMessage`, `NLIPSubMessage`, `AllowedFormat`, `ReservedToken` | Models NLIP messages and provides helpers for text, tokens, structured data, and serialization. |
| `session.py` | `NLIPSession`, `NLIPSessionTurn` | Maintains a session ID, lock, activity time, and bounded request/response history. |
| `nlip_client/` | `NLIPClient`, `NLIPClientSession` | Sends messages to named destinations and maintains independent outbound conversations. |
| `nlip_server/` | `NLIPServer` | Provides the server-side session, security, protocol, and FastAPI pipeline around an agent's `handle()` method. |
| `nlip_security_extensions/` | `SecurityExtension`, `SecurityExtensionManager`, `SecurityEvent`, `SecurityDecision` | Defines and invokes ordered ingress and egress security extensions. |

Import common protocol and transport APIs directly from `nlip_sdk`. See
[`security_extensions/README.md`](../security_extensions/README.md) for instructions on
implementing and enabling an SE.
