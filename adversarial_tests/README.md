# Adversarial Tests

Use this directory for reproducible evaluations of security extensions. Include both
adversarial cases that exercise the target threat and ordinary traffic that measures
false positives or unintended behavior.

A typical evaluation may look like:

```text
adversarial_tests/my_evaluation/
├── attacks.jsonl       # Malicious or policy-violating cases
├── benign.jsonl        # Ordinary comparison cases
├── run_evaluation.py   # Replays cases and records decisions
└── README.md           # Dataset, setup, metrics, and reproduction steps
```

Whenever possible, replay cases through the runnable NLIP application and the configured
enforcement point rather than calling an extension's private implementation directly.
Report what was blocked, replaced, or allowed, along with relevant accuracy, latency, or
cost measurements. The exact file layout and test framework may be changed to fit the
project.
