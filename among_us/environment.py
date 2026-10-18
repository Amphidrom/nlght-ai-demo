"""Game environments behind one port: the simulation and the C# game bridge.

The orchestrator talks to either through ``GameEnvironment``. The bridge speaks
protocol v1 as newline-delimited JSON over a local TCP socket — stdlib on the
Python side, ``TcpListener`` on the C# side, no WebSocket dependency.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from among_us.protocol import (
    ActionResult,
    DecisionRequest,
    Hello,
    MatchEnd,
    Observation,
    Phase,
    ProtocolError,
    SemanticAction,
    decode,
    encode,
)
from among_us.sim.engine import SimEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundTruth:
    """Only the evaluation reads this. No agent ever receives it."""

    roles: dict[str, str]
    events: list[dict[str, Any]]


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

    def ground_truth(self) -> GroundTruth:
        return GroundTruth(
            roles={pid: self.engine.role_of(pid).value for pid in self.engine.player_ids},
            events=[event.to_record() for event in self.engine.events],
        )

    async def close(self) -> None:
        return None


class BridgeEnvironment:
    """Client for the in-game bridge mod (``among_us/bridge``).

    The bridge pushes a ``decision_request`` whenever one of its players reaches
    a decision point and answers each ``action`` with an ``action_result``. The
    game runs in real time; ``advance`` only reports whether it has ended.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 47100, *, decision_wait: float = 1.0) -> None:
        self._host, self._port = host, port
        self._decision_wait = decision_wait
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._decisions: asyncio.Queue[Observation] = asyncio.Queue()
        self._results: dict[str, asyncio.Queue[ActionResult]] = {}
        self._end: MatchEnd | None = None
        self._hello: asyncio.Future[Hello] | None = None
        self._pump: asyncio.Task[None] | None = None

    async def connect(self) -> Hello:
        self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
        self._hello = asyncio.get_running_loop().create_future()
        self._pump = asyncio.create_task(self._read_loop())
        return await asyncio.wait_for(self._hello, timeout=10)

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while line := await self._reader.readline():
                try:
                    message = decode(line)
                except ProtocolError as exc:
                    # A malformed or over-sharing message is dropped loudly;
                    # it never reaches an agent.
                    logger.error("among_us.bridge.protocol_error | %s", exc)
                    continue
                self._dispatch(message)
        finally:
            if self._end is None:
                self._end = MatchEnd("aborted", "bridge disconnected")

    def _dispatch(self, message: Any) -> None:
        if isinstance(message, Hello) and self._hello and not self._hello.done():
            self._hello.set_result(message)
        elif isinstance(message, DecisionRequest):
            self._decisions.put_nowait(message.observation)
        elif isinstance(message, ActionResult):
            self._results.setdefault(message.agent_id, asyncio.Queue()).put_nowait(message)
        elif isinstance(message, MatchEnd):
            self._end = message
        else:
            logger.warning("among_us.bridge.unexpected | %s", type(message).__name__)

    async def next_decisions(self) -> list[Observation]:
        batch: list[Observation] = []
        try:
            batch.append(await asyncio.wait_for(self._decisions.get(), timeout=self._decision_wait))
        except TimeoutError:
            return []
        while not self._decisions.empty():
            batch.append(self._decisions.get_nowait())
        # Only the newest observation per agent is current.
        latest = {obs.agent_id: obs for obs in batch}
        return list(latest.values())

    async def submit(self, action: SemanticAction) -> ActionResult:
        if self._writer is None:
            return ActionResult(action.agent_id, False, "bridge not connected")
        self._writer.write((encode(action) + "\n").encode())
        await self._writer.drain()
        queue = self._results.setdefault(action.agent_id, asyncio.Queue())
        try:
            return await asyncio.wait_for(queue.get(), timeout=5)
        except TimeoutError:
            return ActionResult(action.agent_id, False, "bridge did not answer")

    async def advance(self) -> MatchEnd | None:
        return self._end

    def ground_truth(self) -> GroundTruth | None:
        # The real game's truth is not exported; evaluation is limited to what
        # the decisions themselves record.
        return None

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._pump is not None:
            self._pump.cancel()
