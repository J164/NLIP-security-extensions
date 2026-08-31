"""Process-local NLIP conversation state shared by clients and servers."""

from __future__ import annotations

from asyncio import Lock
from collections import deque
from dataclasses import dataclass, field
from time import monotonic

from .nlip import NLIPMessage


@dataclass(frozen=True, slots=True)
class NLIPSessionTurn:
    """One completed request/response exchange in a conversation."""

    request: NLIPMessage
    response: NLIPMessage


@dataclass(slots=True)
class NLIPSession:
    """A session identity, lock, and bounded NLIP request/response history."""

    id: str
    peer: str | None = None
    max_history: int = 100
    last_active: float = field(default_factory=monotonic, init=False)
    # One conversation is processed in order even when requests arrive concurrently.
    lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _history: deque[NLIPSessionTurn] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("NLIP session id cannot be empty.")
        if self.max_history < 1:
            raise ValueError("NLIP session max_history must be positive.")
        self._history = deque(maxlen=self.max_history)

    @property
    def history(self) -> tuple[NLIPSessionTurn, ...]:
        """Return defensive copies so extensions cannot rewrite canonical history."""
        return tuple(
            NLIPSessionTurn(
                turn.request.model_copy(deep=True),
                turn.response.model_copy(deep=True),
            )
            for turn in self._history
        )

    def record(self, request: NLIPMessage, response: NLIPMessage) -> None:
        """Append one finalized turn and discard the oldest turn at the bound."""
        self._history.append(
            NLIPSessionTurn(
                request.model_copy(deep=True),
                response.model_copy(deep=True),
            )
        )
        self.touch()

    def touch(self) -> None:
        self.last_active = monotonic()
