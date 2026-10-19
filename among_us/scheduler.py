"""Who thinks with the model, in which order, and how many at once.

The environment says *which* agents are at a decision point; the scheduler
decides *how* each one decides. A full lobby can put fifteen agents at a
decision point in the same tick, and a local model serves only a few calls at
a time. So:

- **Priority.** Meetings and votes first, then agents that just saw something
  salient (a kill, a body), then kill opportunities, then routine goals.
- **Concurrency.** At most ``max_concurrent_model_turns`` model turns run at
  once; the rest queue in priority order.
- **Budget.** At most ``model_turns_per_tick`` model turns per tick during the
  task phase. Agents beyond it decide by the deterministic policy this tick —
  the game keeps moving instead of waiting. Meeting and vote turns are exempt:
  they are the decisions the experiment is about.
- **Deadline.** A model turn that exceeds ``turn_timeout_seconds`` — or the
  shorter deadline the game sends with a decision request — is replaced by a
  policy turn, so one slow call cannot stall the match and the answer still
  arrives while the game waits for it.

A policy turn runs the same workflow with the model switched off: the agent
still perceives, remembers and believes; only the decide step is skipped.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from among_us.protocol import Observation, Phase

logger = logging.getLogger(__name__)

TurnFn = Callable[[Observation, bool], Awaitable[dict[str, Any]]]
"""Run one agent turn; the flag says whether the model may be used."""


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    max_concurrent_model_turns: int = Field(default=4, ge=1, le=15, description="Model turns running at once.")
    model_turns_per_tick: int = Field(default=8, ge=0, le=15, description="Model turns per task-phase tick; 0 = policy only outside meetings.")
    turn_timeout_seconds: float = Field(default=90.0, gt=0, le=600, description="Deadline of one model turn.")


def priority(observation: Observation) -> int:
    if observation.phase in (Phase.MEETING, Phase.VOTING):
        return 0
    reason = observation.decision_reason
    if reason.startswith("observed:"):
        return 1
    if reason == "opportunity":
        return 2
    return 3


class DecisionScheduler:
    def __init__(self, config: SchedulerConfig | None = None, *, model_enabled: bool = True) -> None:
        self.config = config or SchedulerConfig()
        self._model_enabled = model_enabled
        self._slots = asyncio.Semaphore(self.config.max_concurrent_model_turns)
        self._spent: Counter[int] = Counter()
        self.stats: Counter[str] = Counter()

    def plan(self, batch: Sequence[Observation]) -> list[tuple[Observation, str]]:
        """Order the batch and assign each turn a mode: ``model`` or ``policy:<why>``."""

        planned = []
        for observation in sorted(batch, key=lambda o: (priority(o), o.agent_id)):
            if not self._model_enabled:
                mode = "policy:offline"
            elif priority(observation) == 0:
                mode = "model"
            elif self._spent[observation.tick] < self.config.model_turns_per_tick:
                self._spent[observation.tick] += 1
                mode = "model"
            else:
                mode = "policy:budget"
            planned.append((observation, mode))
        return planned

    async def run(
        self, batch: Sequence[Observation], turn: TurnFn, *, timeout: float | None = None,
    ) -> list[tuple[Observation, dict[str, Any], str]]:
        """Decide the batch. ``timeout`` (the game's own deadline) shortens ``turn_timeout_seconds``."""

        planned = self.plan(batch)
        deadline = self.config.turn_timeout_seconds if timeout is None else min(timeout, self.config.turn_timeout_seconds)

        async def one(observation: Observation, mode: str) -> tuple[Observation, dict[str, Any], str]:
            if mode != "model":
                self.stats[mode] += 1
                return observation, await turn(observation, False), mode
            async with self._slots:
                try:
                    result = await asyncio.wait_for(turn(observation, True), deadline)
                except TimeoutError:
                    logger.warning("among_us.scheduler.timeout | agent=%s tick=%s", observation.agent_id, observation.tick)
                    self.stats["policy:timeout"] += 1
                    return observation, await turn(observation, False), "policy:timeout"
            self.stats["model"] += 1
            return observation, result, "model"

        # Tasks are created in priority order; the semaphore admits them FIFO.
        return list(await asyncio.gather(*(one(obs, mode) for obs, mode in planned)))
