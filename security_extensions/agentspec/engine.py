"""A small AgentSpec-style rule engine: parse rules, then evaluate them on an event.

Rules use the shape from AgentSpec (Wang, Poskitt, Sun, ICSE '26)::

    rule @block_field_impersonation
    trigger before_flatten
    check from_subagent
          is_reserved_field
          !sender_owns_field
    enforce drop_part
    end

``trigger`` names the event and may name one action (``trigger before_action place_order``).
``check`` is a conjunction of named predicates; ``!`` negates one. A rule with no checks
always fires on its trigger. ``enforce`` lists one or more enforcement names; what each
one does is up to the enforcement point (the SE or the tool guard).

Predicates are plain functions ``(context) -> bool`` registered by name, so a handful of
them can be reused across many rules, which is the point of the AgentSpec design.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Predicate = Callable[[Any], bool]

TRIGGERS = {"before_flatten", "before_action"}
ENFORCEMENTS = {"drop_part", "stop", "user_inspection"}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class RuleSyntaxError(ValueError):
    """A rule file that cannot be parsed or names something unknown."""


@dataclass(frozen=True, slots=True)
class Check:
    predicate: str
    negated: bool = False

    def __str__(self) -> str:
        return ("!" if self.negated else "") + self.predicate


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    trigger: str
    action: str | None
    checks: tuple[Check, ...]
    enforce: tuple[str, ...]

    def applies_to(self, trigger: str, action: str | None = None) -> bool:
        return self.trigger == trigger and (self.action is None or self.action == action)

    def holds(self, context: Any, predicates: Mapping[str, Predicate]) -> bool:
        return all(bool(predicates[c.predicate](context)) != c.negated for c in self.checks)


def parse_rules(text: str, predicates: Iterable[str]) -> list[Rule]:
    """Parse rule text. Every predicate a rule names must be in ``predicates``."""
    known = set(predicates)
    rules: list[Rule] = []
    current: dict[str, Any] | None = None
    section: str | None = None

    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        words = line.split()
        head, rest = words[0], words[1:]

        def fail(message: str) -> RuleSyntaxError:
            return RuleSyntaxError(f"line {number}: {message}")

        if current is None:
            if head != "rule" or len(rest) != 1 or not rest[0].startswith("@"):
                raise fail("expected 'rule @name'")
            name = rest[0][1:]
            if not _NAME.fullmatch(name) or any(r.name == name for r in rules):
                raise fail(f"bad or duplicate rule name {rest[0]!r}")
            current = {"name": name, "trigger": None, "action": None, "checks": [], "enforce": []}
            section = None
            continue

        if head == "end":
            if current["trigger"] is None or not current["enforce"]:
                raise fail(f"rule @{current['name']} needs a trigger and an enforce line")
            rules.append(Rule(current["name"], current["trigger"], current["action"],
                              tuple(current["checks"]), tuple(current["enforce"])))
            current = None
            continue

        if head == "trigger":
            if current["trigger"] is not None or not 1 <= len(rest) <= 2 or rest[0] not in TRIGGERS:
                raise fail(f"expected 'trigger <{'|'.join(sorted(TRIGGERS))}> [action]'")
            current["trigger"] = rest[0]
            current["action"] = rest[1] if len(rest) == 2 else None
            section = None
            continue
        if head in ("check", "enforce"):
            section, words = head, rest
        elif section is None:
            raise fail(f"unexpected {head!r}")
        # Continuation lines extend the last check/enforce section.
        for word in words:
            if section == "check":
                check = Check(word.lstrip("!"), word.startswith("!"))
                if check.predicate not in known:
                    raise fail(f"unknown predicate {check.predicate!r}")
                current["checks"].append(check)
            else:
                if word not in ENFORCEMENTS:
                    raise fail(f"unknown enforcement {word!r}")
                current["enforce"].append(word)

    if current is not None:
        raise RuleSyntaxError(f"rule @{current['name']} has no 'end'")
    return rules


class RuleEngine:
    """Rules plus the predicate vocabulary they are evaluated with."""

    def __init__(self, rules: list[Rule], predicates: Mapping[str, Predicate]) -> None:
        self.rules = rules
        self.predicates = dict(predicates)

    @classmethod
    def from_file(cls, path: str | Path, predicates: Mapping[str, Predicate]) -> "RuleEngine":
        return cls(parse_rules(Path(path).read_text(), predicates), predicates)

    def fired(self, trigger: str, context: Any, action: str | None = None) -> list[Rule]:
        """Every rule on this trigger whose checks all hold, in file order."""
        return [
            rule for rule in self.rules
            if rule.applies_to(trigger, action) and rule.holds(context, self.predicates)
        ]
