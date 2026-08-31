# NLIP Security Extensions

This repository is a course framework for building and evaluating security extensions
for NLIP-enabled agent systems.

## NLIP Introduction

The Natural Language Interaction Protocol (NLIP) is an Ecma-standardized communication
protocol for interactions between users, client programs, and AI agents. It defines a
common structured message format and transport bindings so agents built with different
models or frameworks can communicate without exposing their internal implementations.
NLIP messages can carry natural-language text, structured data, binary data, control
information, and other content types.

Visit the [NLIP project website](https://nlip-project.org/#/) for an overview and the
[Ecma TC56 published standards](https://ecma-international.org/technical-committees/tc56/?tab=published-standards)
for the protocol specifications.

## NLIP Security Extension

An NLIP Security Extension (SE), also called a security function, is a pluggable check at
an agent communication boundary. An SE can inspect an incoming message before an agent
processes it (ingress), or inspect an outgoing message before it leaves the agent
(egress). This makes NLIP a framework-independent enforcement layer for monitoring,
validation, policy enforcement, and defenses against both system-level and semantic
attacks.

In this framework, an SE receives the current NLIP message and session context, then
returns an explicit `ALLOW`, `BLOCK`, or `REPLACE` decision. These are the built-in
decisions; developers may add other decision types and enforcement behavior when their
security scenario requires them. Applications enable ordered SEs for their client and
server traffic through configuration.

## Repository Structure

- [`nlip_sdk/`](nlip_sdk/) contains NLIP message models, client/server communication,
  session tracking, shared configuration loading, and the SE manager and interfaces.
- [`nlip_examples/`](nlip_examples/) contains framework-native agents and a runnable
  example showing how to add NLIP client/server capabilities.
- [`agent_applications/`](agent_applications/) is for runnable, task-oriented,
  NLIP-enabled agent systems used to exercise security extensions.
- [`security_extensions/`](security_extensions/) contains SE implementations, a minimal
  audit example, and a starting location for new extensions.
- [`adversarial_tests/`](adversarial_tests/) is for attack cases, ordinary comparison
  traffic, datasets, and evaluation code.
- [`tests/`](tests/) contains regression tests for the shared framework.

Start with [`nlip_examples/README.md`](nlip_examples/README.md) to understand the agent
integration flow, then read
[`security_extensions/README.md`](security_extensions/README.md) for the SE interface,
registration, and enabling model.

## SE Development Guidelines

Fork this repository and create a dedicated branch for developing and testing your
security extension. Ideally, a contribution should include:

1. A runnable NLIP-enabled application under `agent_applications/`, including any agent,
   task environment, tools, client/server integration, and configuration needed for the
   scenario.
2. A pluggable implementation under `security_extensions/`, enabled at the appropriate
   ingress and/or egress points in the application's configuration.
3. Adversarial and ordinary test cases under `adversarial_tests/` that demonstrate the
   attack, evaluate the defense, and make the results reproducible.

Document how to install dependencies, run the application, enable the SE, and reproduce
the evaluation. Keep application-specific behavior out of the shared SDK unless it is
needed by more than one application or extension.

**Note for developers:** You do not have to follow this framework rigidly. If it limits
your security idea, agent application, or evaluation scenario, you may change the
directory structure, replace components, or add and remove framework code as needed.

## Known Limitation

The shared SE framework automatically covers NLIP ingress and egress, but it cannot
automatically observe an agent's internal tool calls or tool results. To enforce security
at this boundary, developers must connect their SE logic to the hooks provided by the
agent framework they use.

## License

This project is licensed under the [MIT License](LICENSE).
