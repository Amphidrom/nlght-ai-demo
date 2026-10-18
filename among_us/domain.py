"""Objective game events: the ground truth only replay and evaluation read."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from among_us.protocol import ObservedEvent, to_dict


@dataclass(frozen=True)
class GameEvent:
    """What actually happened, and exactly what each witness perceived of it.

    ``views`` is the only bridge from the truth to an agent: an agent receives
    its own view of an event and nothing else. A fake task, for example, is a
    ``FAKE_TASK`` here and a plain ``TASK_SEEN`` in every view.
    """

    event_id: str
    tick: int
    kind: str
    room: str | None = None
    actor: str | None = None
    target: str | None = None
    text: str | None = None
    views: dict[str, ObservedEvent] = field(default_factory=dict)

    @property
    def witnesses(self) -> frozenset[str]:
        return frozenset(self.views)

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tick": self.tick,
            "kind": self.kind,
            "room": self.room,
            "actor": self.actor,
            "target": self.target,
            "text": self.text,
            "views": {agent: to_dict(view) for agent, view in sorted(self.views.items())},
        }
