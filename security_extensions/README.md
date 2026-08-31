# Security Extensions

Security extensions inspect NLIP messages at two stable points:

- `INGRESS`: a message enters this entity. This is a server request or a client
  response.
- `EGRESS`: a message leaves this entity. This is a server response or a client
  request.

`SecurityEvent.component` says whether that event came from the entity's `client` or
`server` side; it does not create four different enforcement-point APIs. The event also
contains the current message, local entity name, peer when known, session, completed
prior session history, and the paired request for response events.

An extension implements one async method and must return an explicit decision:

```python
from nlip_sdk.nlip_security_extensions import (
    SecurityDecision,
    SecurityEvent,
    SecurityExtension,
)


class MySecurityExtension(SecurityExtension):
    async def evaluate(self, event: SecurityEvent) -> SecurityDecision:
        return SecurityDecision.allow()
```

The other decisions are `SecurityDecision.block(reason)` and
`SecurityDecision.replace(message, reason)`. Extensions at one point run in configured
order; a replacement is passed to the next extension, while a block stops immediately.
An exception also fails closed instead of silently bypassing the extension.

## Registration and enabling

Definitions and per-entity enabling are separate in `config.toml`:

```toml
[security_extensions.audit_example]
entrypoint = "security_extensions.simple_example.audit_security_extension:AuditSecurityExtension"

[entities.claude.security]
ingress = ["audit_example"]
egress = ["audit_example"]
```

The `module:Class` entrypoint is imported when the entity starts. Because this executes
Python code, the configuration and extension packages must be trusted. Presence in a
list means enabled, and list order is execution order. Optional constructor arguments
can be placed under `[security_extensions.<name>.options]`.

`simple_example/audit_security_extension.py` implements `AuditSecurityExtension`,
which prints the current message and its current-turn context, but not prior session
history, and always returns `ALLOW`. It is enabled for both
agents in the integration demo. Client-side extensions are async, so use
`send()`/`send_text()` on the asynchronous client API.
The output can contain message content and NLIP tokens, so this example logger is not
appropriate for production or untrusted logs.

## Tool and environment interactions

The framework above observes messages crossing NLIP client and server boundaries. It
does not automatically observe an agent's internal interaction with its external
environment, including a tool call or the tool result returned to the agent. To inspect
or enforce at those boundaries, add framework-specific detection points to the agent
application and invoke the relevant security function there.

For example, the Python Claude Agent SDK provides these tool-related hooks:

- `PreToolUse`: inspect, deny, or modify a requested tool call before execution.
- `PostToolUse`: inspect or modify a successful tool result before Claude consumes it.
- `PostToolUseFailure`: observe and handle a failed tool execution.
- `PermissionRequest`: make a custom permission decision when a tool requires approval.

Claude hooks can be registered through `ClaudeAgentOptions.hooks` and filtered by tool
name with `HookMatcher`, including MCP tool names such as
`mcp__<server>__<action>`. These hooks belong to the Claude Agent SDK and are not
registered by `SecurityExtensionManager`; an application developer must connect their
security logic to the appropriate hooks. See the
[Claude Agent SDK hooks documentation](https://code.claude.com/docs/en/agent-sdk/hooks)
for the current hook inputs and decisions. Other agent frameworks require their own
equivalent integration points.
