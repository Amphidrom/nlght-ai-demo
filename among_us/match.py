"""Match orchestration: route decision points to agents and actions to the game.

An agent turn is the ``among_us_agent_turn`` workflow. In the running demo it
goes through the nlght-ai executor (``WorkflowAgentRunner``), with one session
per agent and match. ``PipelineAgentRunner`` runs the same steps in process —
for tests and for offline matches without a database or model.

The match record keeps the objective event log apart from the agents'
decisions; only evaluation joins the two.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from among_us.environment import GameEnvironment, GroundTruth
from among_us.protocol import (
    ActionKind,
    Observation,
    Phase,
    ProtocolError,
    SemanticAction,
    action_from_dict,
    to_dict,
)
from among_us.steps import STEPS
from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.core.entry.context import RequestContext
from nlght.core.hive_mind.models import TurnSummary
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext
from trigger.invocation import build_invocation

logger = logging.getLogger(__name__)

WORKFLOW = "among_us_agent_turn"


def session_key(match_id: str, agent_id: str) -> str:
    return f"among-us:{match_id}:{agent_id}"


def _context(correlation_id: str) -> RequestContext:
    return RequestContext(
        correlation_id=correlation_id,
        request_id=str(uuid.uuid4()),
        received_at=datetime.now(UTC),
        path="/among-us/internal/agent-turn",
        method="POST",
        headers={},
        query_params={},
        client_host=None,
    )


class AgentRunner(Protocol):
    async def decide(self, *, agent_id: str, session_key: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class WorkflowAgentRunner:
    """One agent turn = one run of the seeded workflow, in the agent's own session."""

    def __init__(self, container: Any, *, model: str | None = None) -> None:
        self._container = container
        self._model = model

    async def decide(self, *, agent_id: str, session_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        cid = str(uuid.uuid4())
        body = {**payload, **({"model": self._model} if self._model else {})}
        invocation = await build_invocation(
            self._container, workflow_name=WORKFLOW, payload=body,
            context=_context(cid), session_key=session_key, stream=False,
        )
        executor = self._container.workflow_executor
        content = ""
        if hasattr(executor, "stream_signals"):
            async for signal in executor.stream_signals(invocation):
                if signal.kind == "result" and signal.content:
                    content = signal.content
        else:
            response = await executor.execute(invocation)
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _parse_result(content)


class InMemorySession:
    """The coordinator surface the agent steps use, kept in process memory."""

    def __init__(self) -> None:
        self.slots: dict[str, Any] = {}
        self.turns: list[TurnSummary] = []

    def get_slot(self, key: str, default_factory: Callable[[], Any]) -> Any:
        if key not in self.slots:
            return default_factory()
        return json.loads(json.dumps(self.slots[key]))

    def set_slot(self, key: str, value: object) -> None:
        # Round-trip through JSON, as Hive Mind persistence would.
        self.slots[key] = json.loads(json.dumps(value))

    def next_turn_nr(self) -> int:
        return len(self.turns) + 1

    def record_turn(self, turn: TurnSummary) -> None:
        self.turns.append(turn)


class PipelineAgentRunner:
    """Runs the five agent steps directly, one ``InMemorySession`` per session key."""

    def __init__(self, llm: Any = None, *, config: dict[str, dict[str, Any]] | None = None) -> None:
        self._llm = llm
        self._config = config or {}
        self.sessions: dict[str, InMemorySession] = {}

    async def decide(self, *, agent_id: str, session_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        cid = str(uuid.uuid4())
        emitter = BufferingSignalEmitter()
        ctx = WorkflowStepContext(
            correlation_id=cid,
            trigger=Trigger(
                kind=TriggerKind.HTTP_REQUEST, protocol=ProtocolKind.GENERIC_JSON, operation=WORKFLOW,
                payload=payload, context=_context(cid), stream=False, session_key=session_key,
            ),
            model="",
            messages=[],
            stream=False,
            emitter=emitter,
            llm=self._llm,
            store_coordinator=self.sessions.setdefault(session_key, InMemorySession()),  # type: ignore[arg-type]
        )
        for step_cls in STEPS:
            result = await step_cls(config=self._config.get(step_cls.TYPE, {})).run(ctx)
            ctx = result.ctx
            if result.verdict in ("done", "failed"):
                break
        content = next((s.content for s in reversed(emitter.collected()) if s.kind == "result"), "")
        return _parse_result(content)


def _parse_result(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content) if content else {}
    except json.JSONDecodeError:
        return {"error": "agent returned no JSON result"}
    return value if isinstance(value, dict) else {"error": "agent result is not an object"}


@dataclass
class MatchRecord:
    match_id: str
    seed: int
    egos: dict[str, str]
    environment: str
    status: str = "pending"
    winner: dict[str, Any] | None = None
    error: str | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)
    ground_truth: GroundTruth | None = None
    _changed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def append(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)
        self._notify()

    def set_status(self, status: str, **values: Any) -> None:
        self.status = status
        for key, value in values.items():
            setattr(self, key, value)
        self._notify()

    def _notify(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    async def wait_for_change(self, timeout: float = 15.0) -> None:
        event = self._changed
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            pass

    def summary(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "seed": self.seed,
            "egos": self.egos,
            "environment": self.environment,
            "status": self.status,
            "winner": self.winner,
            "error": self.error,
            "decisions": sum(1 for e in self.entries if e["type"] == "decision"),
        }


class ReplayWriter:
    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)

    def write(self, record: MatchRecord) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{record.match_id}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "match", **record.summary()}) + "\n")
            for entry in record.entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if record.ground_truth is not None:
                handle.write(json.dumps({"type": "ground_truth", "roles": record.ground_truth.roles}) + "\n")
                for event in record.ground_truth.events:
                    handle.write(json.dumps({"type": "event", **event}, ensure_ascii=False) + "\n")
        return path


async def run_match(
    environment: GameEnvironment,
    runner: AgentRunner,
    record: MatchRecord,
    *,
    offline: bool = False,
    max_iterations: int = 5000,
) -> MatchRecord:
    record.set_status("running")
    try:
        hello = await environment.connect()
        players = list(hello.players)
        missing = [p for p in players if p not in record.egos]
        if missing:
            raise ValueError(f"no ego assigned to players {missing}")
        record.append({"type": "hello", **to_dict(hello)})
        for _ in range(max_iterations):
            for _ in range(len(players) * 3):
                batch = await environment.next_decisions()
                if not batch:
                    break
                await _decide_batch(environment, runner, record, batch, players, hello.source, offline)
            end = await environment.advance()
            if end is not None:
                record.set_status("finished", winner=to_dict(end))
                break
        else:
            record.set_status("failed", error="iteration limit reached")
    except Exception as exc:
        logger.exception("among_us.match.failed | match=%s", record.match_id)
        record.set_status("failed", error=str(exc))
    finally:
        record.ground_truth = environment.ground_truth()
        await environment.close()
    return record


async def _decide_batch(
    environment: GameEnvironment,
    runner: AgentRunner,
    record: MatchRecord,
    batch: Sequence[Observation],
    players: list[str],
    source: str,
    offline: bool,
) -> None:
    async def turn(observation: Observation) -> dict[str, Any]:
        payload = {
            "match_id": record.match_id,
            "match_seed": record.seed,
            "agent_id": observation.agent_id,
            "ego": record.egos[observation.agent_id],
            "players": players,
            "source": source,
            "offline": offline,
            "observation": to_dict(observation),
        }
        return await runner.decide(
            agent_id=observation.agent_id,
            session_key=session_key(record.match_id, observation.agent_id),
            payload=payload,
        )

    # Independent agents think in parallel; the game applies their actions
    # in seat order so the outcome does not depend on model latency.
    results = await asyncio.gather(*(turn(obs) for obs in batch))
    for observation, result in sorted(zip(batch, results, strict=True), key=lambda pair: pair[0].agent_id):
        idle = SemanticAction(
            observation.agent_id,
            ActionKind.SKIP_VOTE if observation.phase is Phase.VOTING else ActionKind.WAIT,
        )
        try:
            action = action_from_dict(result["action"])
        except (KeyError, ProtocolError, TypeError):
            action = idle
        if action.agent_id != observation.agent_id:
            action = idle
        outcome = await environment.submit(action)
        if not outcome.accepted:
            # The game refused it (it may have moved on since the observation):
            # stand still rather than retry a stale decision.
            await environment.submit(idle)
        record.append({
            "type": "decision",
            "agent_id": observation.agent_id,
            "ego": record.egos[observation.agent_id],
            "tick": observation.tick,
            "phase": observation.phase.value,
            "reason": observation.decision_reason,
            "perceived": [e.event_id for e in observation.events],
            "result": result,
            "action": to_dict(action),
            "accepted": outcome.accepted,
            "rejection": outcome.reason,
        })
