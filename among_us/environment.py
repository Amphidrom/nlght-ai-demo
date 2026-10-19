"""The simulated game behind the ``GameEnvironment`` port.

``run_match`` drives a match by pulling decision points from an environment.
The real game does not fit that shape — it runs in real time and pushes its
decision points — so it is reached through lobby sessions instead
(``among_us.lobby``), which feed the same agent turns.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from among_us.protocol import (
    ActionResult,
    Hello,
    MatchEnd,
    Observation,
    Phase,
    SemanticAction,
)
from among_us.sim.engine import SimEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundTruth:
    """Only the evaluation reads this. No agent ever receives it."""

    roles: dict[str, str]
    events: list[dict[str, Any]]
    #: tick → player → room. Empty where the environment cannot provide it.
    positions: dict[int, dict[str, str]] = field(default_factory=dict)


class GameEnvironment(Protocol):
    async def connect(self) -> Hello: ...

    async def next_decisions(self) -> list[Observation]:
        """Agents at a decision point now, each with its own observation."""
        ...

    async def submit(self, action: SemanticAction) -> ActionResult: ...

    async def advance(self) -> MatchEnd | None:
        """Let the game move on; the match result once it has ended."""
        ...

    def ground_truth(self) -> GroundTruth | None: ...

    async def close(self) -> None: ...


class SimEnvironment:
    def __init__(self, players: Sequence[str], *, seed: int, **options: Any) -> None:
        self.engine = SimEngine(players, seed=seed, **options)

    async def connect(self) -> Hello:
        return Hello(self.engine.game_version, self.engine.source, self.engine.player_ids)

    async def next_decisions(self) -> list[Observation]:
        waiting = [pid for pid in self.engine.alive() if self.engine.needs_decision(pid)]
        if self.engine.phase is Phase.MEETING:
            # Speakers take turns, so each one hears everything said before.
            waiting = waiting[:1]
        return [self.engine.observe(pid) for pid in waiting]

    async def submit(self, action: SemanticAction) -> ActionResult:
        return self.engine.submit(action)

    async def advance(self) -> MatchEnd | None:
        return self.engine.advance()

    def spectator_view(self) -> dict[str, Any]:
        return self.engine.spectator_view()

    def ground_truth(self) -> GroundTruth:
        return GroundTruth(
            roles={pid: self.engine.role_of(pid).value for pid in self.engine.player_ids},
            events=[event.to_record() for event in self.engine.events],
            positions={tick: dict(rooms) for tick, rooms in self.engine.positions.items()},
        )

    async def close(self) -> None:
        return None
