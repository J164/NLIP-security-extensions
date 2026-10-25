# AgentSpec rules for NLIP

Runtime rules in the style of AgentSpec (Wang, Poskitt, Sun, ICSE '26) for the shopping
assistant's orchestrator. They have two enforcement points:

| Piece | Where it runs | What it sees |
|---|---|---|
| `extension.py`, `AgentSpecExtension` | orchestrator NLIP ingress, client side (replies from subagents) | each part of a reply, before it is merged into order state or rendered into the model's prompt |
| `guard.py`, `AgentSpecToolGuard` | orchestrator tool boundary (`ToolGuard.before_tool_call`) | the tool call, the user's OAuth scope, the order state |

User traffic (server-side ingress) is not checked: a malicious user is out of scope.

## Rules (`rules.spec`)

| Rule | Trigger | Checks | Enforce | Threat |
|---|---|---|---|---|
| `@block_label_spoofing` | `before_flatten` | `from_subagent`, `!is_token`, `is_privileged_label` | `drop_part` | a part labeled as a chat role (`user`, `system`, …) or an agent |
| `@block_malformed_label` | `before_flatten` | `from_subagent`, `!is_token`, `!is_wellformed_label` | `drop_part` | a label with line breaks, control characters or prompt syntax (`[ ] ( ) =`, spaces) that could forge `[from <agent>]` lines |
| `@block_forged_provenance` | `before_flatten` | `from_subagent`, `!is_token`, `forges_provenance` | `drop_part` | text content that starts a new `[from <agent>]` line |
| `@block_field_impersonation` | `before_flatten` | `from_subagent`, `is_reserved_field`, `!sender_owns_field` | `drop_part` | a subagent writing a field another entity owns |
| `@enforce_purchase_scope` | `before_action place_order` | `exceeds_max_purchase`, `purchase_authorized` | `stop` | order state authorizes an order above the OAuth limit |
| `@require_confirmation_above_threshold` | `before_action place_order` | `at_or_above_confirm_threshold`, `confirmation_waived` | `user_inspection` | order state waives a confirmation the OAuth scope requires |

A label is well formed when it matches `[A-Za-z][A-Za-z0-9_.-]{0,63}`. Privileged labels
match exactly, ignoring case and surrounding spaces, so `user_rating` is not `user`.

Enforcements:

- `drop_part`: remove that part and deliver the rest of the reply (an SE `REPLACE`), so the
  honest search results or reviews still reach the model. A dropped primary part is
  replaced with `[removed by security policy]`; tokens are always kept.
- `stop`: at ingress, block the whole reply (`BLOCK`, which fails the user's turn with
  HTTP 403); at the tool boundary, refuse the call.
- `user_inspection`: the order becomes pending until the user presses Confirm.

### Provenance tag

After the rules run, the SE appends one `provenance` part to every subagent reply (option
`tag_provenance`, default on). It names the sender (the endpoint the orchestrator dialed),
lists every remaining part with its label and format, lists every removed part with the rule
that removed it, and says that none of the reply comes from the user or another agent. The
orchestrator renders it into the prompt like any other part:

```
[from review_agent] provenance (structured/json) = {"note": "Every part of this reply was sent by review_agent. ...", "parts": [...], "removed": [{"label": "requires_confirmation", "part": 2, "rule": "@block_field_impersonation"}], "sender": "review_agent", ...}
```

`provenance` is a reserved field owned by the orchestrator, so a subagent that sends its own
is caught by `@block_field_impersonation`, and a malformed label is listed as
`<malformed label>` rather than echoed. Tagging is logged as action `tag`, which the
benchmark does not count as a detection or a false positive.

### What the rules do not catch

A subagent that lies inside a field it owns (the review agent inflating its own
`review_summary`) passes every rule: provenance is correct, only the content is false. The
`owned_field_lie` attack cases measure this. Content checks (e.g. recomputing ratings from
the review data) would need data the orchestrator does not have.

The two tool-boundary rules fire only when order state disagrees with the OAuth scope. When
state already refuses or already asks for confirmation, the app does the right thing by
itself and the rule stays silent, so it doesn't count as a false positive.

## Predicates (`predicates.py`)

Twelve predicates across six rules. The sender (`provenance`) is `SecurityEvent.peer`: the
endpoint the orchestrator itself dialed, not anything the message says about itself. Field
owners, roles, agent names and the label pattern are options with defaults (below), so the
policy is data and not code.

## Enabling

```toml
[security_extensions.agentspec]
entrypoint = "security_extensions.agentspec.extension:AgentSpecExtension"
# [security_extensions.agentspec.options]
# rules = "path/to/other.spec"        # default: rules.spec next to the code
# roles = ["system", "user", ...]
# field_owners = { cart_total = "checkout_agent", ... }

[entities.orchestrator.security]
ingress = ["agentspec"]

[tool_guards.agentspec_guard]
entrypoint = "security_extensions.agentspec.guard:AgentSpecToolGuard"

[entities.orchestrator]
tool_guards = ["agentspec_guard"]
```

`adversarial_tests/agentspec_benchmark/configs/defense_on.toml` does exactly this. If
`$SHOP_SECURITY_LOG` is set, every decision (including `allow`) is appended to it as one JSON
row `{"ts", "entity", "extension", "point", "action", "rule", "reason", "ms"}`, which the
benchmark reads for detection and overhead.

## Adding a rule

1. If no existing predicate expresses the check, add a function `(context) -> bool` to
   `predicates.py` and list it in `PREDICATES`. Prefer a small, reusable predicate over one
   that encodes a whole rule.
2. Add the rule to `rules.spec`. The parser rejects unknown predicates, triggers and
   enforcements, so a typo fails at startup rather than silently disabling the rule.
3. Add a unit test in `tests/test_agentspec_rules.py` and, if an attack case should trigger
   it, set that case's `expected_rule`. `pytest tests/` then checks offline that every
   attack case fires its expected rule and that no benign case fires any rule.
