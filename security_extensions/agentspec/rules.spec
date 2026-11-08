# AgentSpec rules for the NLIP shopping assistant's orchestrator.
#
# before_flatten: one part of a subagent's reply, before the orchestrator merges it into
#   order state or renders it into the model's prompt. drop_part removes that part and
#   delivers the rest of the reply; stop would block the whole reply.
# before_action <tool>: a tool call the model wants to make (see the note at the end).
#   stop refuses the call; user_inspection makes the user press Confirm first.

# Threat 1: a subagent's part claims a chat role or another agent's identity.
rule @block_label_spoofing
trigger before_flatten
check from_subagent
      !is_token
      is_privileged_label
enforce drop_part
end

# Threat 1, prompt-level form: a label that is not a plain identifier can carry line
# breaks and forge whole "[from <agent>]" lines in the orchestrator's prompt.
rule @block_malformed_label
trigger before_flatten
check from_subagent
      !is_token
      !is_wellformed_label
enforce drop_part
end

# Threat 1, same forgery through content: text that starts a new "[from <agent>]" line.
rule @block_forged_provenance
trigger before_flatten
check from_subagent
      !is_token
      forges_provenance
enforce drop_part
end

# Threat 2: a subagent writes a field another entity owns (confused deputy).
rule @block_field_impersonation
trigger before_flatten
check from_subagent
      is_reserved_field
      !sender_owns_field
enforce drop_part
end

# No before_action rules ship: the orchestrator itself applies the user's scope to the
# quote (agent_applications/shopping_assistant/policy.py), so a tool-boundary scope rule
# would only repeat it. The trigger and the action predicates (exceeds_max_purchase,
# at_or_above_confirm_threshold) remain for rules added later; enable the tool guard in the
# config to evaluate them.
