"""Lobbies of the real game: the nlght Impostor server registers its bots here.

A human hosts the lobby. When they type ``!bots N`` the server registers the
lobby (``LobbyRegistration``); the registry picks N free egos, names the agents
after them and opens the lobby's own session with its own token. The server
spawns exactly those agents. Everything else travels over that session:

- the server posts ``game_started``, ``observation``, ``decision_request``,
  ``action_result``, ``agent_left`` and ``game_ended``;
- the registry streams ``action`` back — answering a request, or on the
  agent's own initiative (``request_id`` null): after an observation the
  agent's beliefs may call for it (``initiative.py``). The server checks it
  against the game as it is then.

A lobby outlives its games: every ``game_started`` opens a new match, and an
agent's memory lives in the session of that match, so it starts fresh with
every game. A token addresses exactly one lobby; no message reaches another.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import random
import secrets
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from among_us.ego import ego_ids, load_ego
from among_us.match import AgentRunner, MatchRecord, chosen_action, decision_entry, session_key
from among_us.models import ModelCatalog, ModelChoice
from among_us.protocol import (
    ActionKind,
    ActionMessage,
    ActionResult,
    AgentLeft,
    DecisionRequest,
    EventKind,
    GameEnded,
    GameStarted,
    LobbyRegistered,
    LobbyRegistration,
    Observation,
    ObservationUpdate,
    Phase,
    SeatedAgent,
    SemanticAction,
    encode,
    to_dict,
)

logger = logging.getLogger(__name__)

#: Players in one Among Us lobby.
MAX_AGENTS = 15
#: Most recent recalled memories kept per agent for the admin UI.
RECALLED_SHOWN = 30
#: Events that tell an observer someone died; the agent named as target is dead.
_DEATHS = frozenset({EventKind.KILL_WITNESSED, EventKind.BODY_FOUND, EventKind.EJECTED, EventKind.KILL_COMMITTED})


class LobbyError(ValueError):
    """A registration or message the registry refuses; ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class LobbyAgent:
    seat: SeatedAgent
    present: bool = True
    #: From the agent's own latest observation; the truth for everyone arrives with game_ended.
    role: str | None = None
    special_role: str | None = None
    alive: bool = True
    room: str | None = None
    last_action: dict[str, Any] | None = None
    last_latency_ms: int | None = None
    #: decisions by how they were made ("model", "policy:timeout", …), and answers the game refused.
    decided: Counter[str] = field(default_factory=Counter)
    refused: int = 0
    #: actions on the agent's own initiative: "sent", "accepted", "refused".
    initiatives: Counter[str] = field(default_factory=Counter)
    #: The operator's models for this agent (admin UI); None: the ego's or the platform default.
    model_override: ModelChoice | None = None
    meeting_override: ModelChoice | None = None
    #: The models its game and meeting turns run on, resolved by the registry.
    model: ModelChoice | None = None
    model_usable: bool = True
    meeting_model: ModelChoice | None = None
    meeting_model_usable: bool = True
    #: What the agent believes and recalls after its latest turn, and its memory statistics.
    beliefs: dict[str, Any] = field(default_factory=dict)
    recalled: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    #: Milliseconds per step of its latest decision, and its plan and stance then.
    timings: dict[str, float] = field(default_factory=dict)
    stance: dict[str, Any] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    pending: asyncio.Task[None] | None = field(default=None, repr=False)

    def summary(self) -> dict[str, Any]:
        return {
            **to_dict(self.seat),
            "present": self.present,
            "role": self.role,
            "special_role": self.special_role,
            "alive": self.alive,
            "room": self.room,
            "last_action": self.last_action,
            "last_latency_ms": self.last_latency_ms,
            "decided": dict(self.decided),
            "refused": self.refused,
            "initiatives": dict(self.initiatives),
            "model": self.model.to_dict() if self.model else None,
            "model_usable": self.model_usable,
            "meeting_model": self.meeting_model.to_dict() if self.meeting_model else None,
            "meeting_model_usable": self.meeting_model_usable,
            "beliefs": self.beliefs,
            "recalled": self.recalled,
            "memory": {k: (len(v) if isinstance(v, list) else v) for k, v in self.memory.items()},
            "timings": self.timings,
            "stance": self.stance,
        }


@dataclass
class Lobby:
    lobby_id: str
    game_code: str
    game_version: str
    source: str
    _token: str = field(repr=False)
    agents: dict[str, LobbyAgent] = field(default_factory=dict)
    state: str = "lobby"
    match: MatchRecord | None = None
    match_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    outbox: asyncio.Queue[str] = field(default_factory=asyncio.Queue, repr=False)
    #: request_id → index of its decision in the match log, to record the game's verdict on it.
    _decisions: dict[str, int] = field(default_factory=dict, repr=False)
    #: agent → indexes of its initiatives the game has not judged yet, oldest first.
    _initiatives: dict[str, list[int]] = field(default_factory=dict, repr=False)
    _changed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def authorized(self, token: str) -> bool:
        return hmac.compare_digest(self._token.encode(), token.encode())

    def notify(self) -> None:
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
            "lobby_id": self.lobby_id,
            "game_code": self.game_code,
            "game_version": self.game_version,
            "source": self.source,
            "state": self.state,
            "match_id": self.match.match_id if self.match else None,
            "matches": list(self.match_ids),
            "created_at": self.created_at,
            "agents": [agent.summary() for agent in self.agents.values()],
        }


def _model_for(agent: LobbyAgent, observation: Observation) -> tuple[ModelChoice | None, bool]:
    if observation.phase in (Phase.MEETING, Phase.VOTING):
        return agent.meeting_model, agent.meeting_model_usable
    return agent.model, agent.model_usable


class LobbyRegistry:
    def __init__(
        self,
        runner: AgentRunner,
        *,
        offline: bool = False,
        catalog: ModelCatalog | None = None,
        on_match_started: Callable[[MatchRecord], None] | None = None,
        on_match_finished: Callable[[MatchRecord], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._runner = runner
        self._offline = offline
        self._catalog = catalog
        self._on_started = on_match_started or (lambda record: None)
        self._on_finished = on_match_finished or (lambda record: None)
        self._rng = rng or random.Random()
        self._tasks: set[asyncio.Task[Any]] = set()
        self.lobbies: dict[str, Lobby] = {}

    @property
    def catalog(self) -> ModelCatalog | None:
        return self._catalog

    # -- registration ---------------------------------------------------------

    def register(self, registration: LobbyRegistration) -> LobbyRegistered:
        lobby = Lobby(
            lobby_id=uuid.uuid4().hex[:12],
            game_code=registration.game_code,
            game_version=registration.game_version,
            source=registration.source,
            _token=secrets.token_urlsafe(32),
        )
        seated = self._seat(lobby, registration.count, registration.players)
        self.lobbies[lobby.lobby_id] = lobby
        logger.info("among_us.lobby.registered | lobby=%s | game=%s | agents=%s",
                    lobby.lobby_id, lobby.game_code, [a.agent_id for a in seated])
        return LobbyRegistered(lobby.lobby_id, lobby._token, seated)

    def add_agents(self, lobby: Lobby, count: int, players: tuple[str, ...]) -> tuple[SeatedAgent, ...]:
        if lobby.state == "closed":
            raise LobbyError("The lobby is closed.", 409)
        return self._seat(lobby, count, players)

    def _seat(self, lobby: Lobby, count: int, players: tuple[str, ...]) -> tuple[SeatedAgent, ...]:
        """Pick ``count`` egos nobody in the lobby uses and no player's name collides with."""

        present = sum(1 for agent in lobby.agents.values() if agent.present)
        if not 1 <= count <= MAX_AGENTS - present:
            raise LobbyError(f"count must be 1 to {MAX_AGENTS - present} for this lobby.")
        taken = {name.strip().lower() for name in players} | set(lobby.agents)
        free = [ego for ego in ego_ids() if ego not in taken and load_ego(ego).name.lower() not in taken]
        if len(free) < count:
            raise LobbyError(f"Only {len(free)} egos are free in this lobby.", 409)
        seated = []
        for ego in sorted(self._rng.sample(free, count)):
            name = load_ego(ego).name
            seat = SeatedAgent(agent_id=name.lower(), name=name, ego=ego)
            agent = LobbyAgent(seat)
            self._resolve_model(agent)
            lobby.agents[seat.agent_id] = agent
            seated.append(seat)
        lobby.notify()
        return tuple(seated)

    def set_model(self, lobby: Lobby, agent_id: str, choice: ModelChoice | None, *, meeting: bool = False) -> LobbyAgent:
        """The operator's model for one agent's game or meeting turns (None: back to the default); from its next turn on."""

        agent = self._agent(lobby, agent_id)
        if meeting:
            agent.meeting_override = choice
        else:
            agent.model_override = choice
        self._resolve_model(agent)
        lobby.notify()
        logger.info("among_us.lobby.model | lobby=%s | agent=%s | %s", lobby.lobby_id, agent_id, agent.model)
        return agent

    def _resolve_model(self, agent: LobbyAgent) -> None:
        if self._catalog is None:
            agent.model, agent.model_usable = agent.model_override, True
            agent.meeting_model, agent.meeting_model_usable = agent.meeting_override or agent.model_override, True
            return
        ego = load_ego(agent.seat.ego)
        agent.model = self._catalog.resolve(ego, agent.model_override)
        if agent.meeting_override is not None:
            agent.meeting_model = agent.meeting_override
        elif ego.model is not None and ego.model.meeting is not None:
            agent.meeting_model = self._catalog.resolve(ego, meeting=True)
        else:
            agent.meeting_model = agent.model   # no meeting model of its own: the game model weighs there too
        agent.model_usable = self._catalog.usable(agent.model)
        agent.meeting_model_usable = self._catalog.usable(agent.meeting_model)

    def lobby(self, lobby_id: str, token: str | None = None) -> Lobby:
        """The lobby; with ``token``, only if the token is that lobby's."""

        lobby = self.lobbies.get(lobby_id)
        if lobby is None:
            raise LobbyError("Lobby not found.", 404)
        if token is not None and not lobby.authorized(token):
            raise LobbyError("Not this lobby's token.", 403)
        return lobby

    def close(self, lobby: Lobby, reason: str = "closed") -> None:
        if lobby.state == "closed":
            return
        for agent in lobby.agents.values():
            if agent.pending is not None:
                agent.pending.cancel()
        if lobby.match is not None:
            self._finish(lobby, status="failed", error=f"lobby {reason}")
        lobby.state = "closed"
        lobby.notify()
        logger.info("among_us.lobby.closed | lobby=%s | reason=%s", lobby.lobby_id, reason)

    # -- the session ---------------------------------------------------------

    def handle(self, lobby: Lobby, message: Any) -> None:
        """One message the server posted. Agent turns run in the background; this returns at once."""

        if lobby.state == "closed":
            raise LobbyError("The lobby is closed.", 409)
        if isinstance(message, GameStarted):
            self._start_game(lobby, message)
        elif isinstance(message, ObservationUpdate):
            agent = self._playing(lobby, message.observation)
            self._spawn(self._observe(lobby, agent, message.observation))
        elif isinstance(message, DecisionRequest):
            agent = self._playing(lobby, message.observation)
            if agent.pending is not None and not agent.pending.done():
                # The game replaced the request it waited for: its answer would be refused anyway.
                agent.pending.cancel()
            agent.pending = self._spawn(self._decide(lobby, agent, message, time.monotonic()))
        elif isinstance(message, ActionResult):
            self._record_result(lobby, message)
        elif isinstance(message, AgentLeft):
            self._agent(lobby, message.agent_id).present = False
        elif isinstance(message, GameEnded):
            if lobby.match is None:
                raise LobbyError("No game is running in this lobby.", 409)
            lobby.match.roles = dict(message.roles)
            self._finish(lobby, status="finished", winner={"winner": message.winner, "reason": message.reason})
        else:
            raise LobbyError(f"{type(message).__name__} is not a message the server sends.")
        lobby.notify()

    def push_action(self, lobby: Lobby, action: SemanticAction, reason: dict[str, Any] | None = None) -> None:
        """An action on the agent's own initiative; the server checks it against the game as it is now."""

        agent = self._agent(lobby, action.agent_id)
        agent.initiatives["sent"] += 1
        if lobby.match is not None:
            lobby._initiatives.setdefault(action.agent_id, []).append(len(lobby.match.entries))
            lobby.match.append({"type": "initiative", "agent_id": action.agent_id, "action": to_dict(action),
                                "reason": reason, "accepted": None, "rejection": None})
        self._send(lobby, ActionMessage(None, action))

    def _start_game(self, lobby: Lobby, message: GameStarted) -> None:
        if lobby.match is not None:
            self._finish(lobby, status="failed", error="a new game started before this one ended")
        egos = {aid: agent.seat.ego for aid, agent in lobby.agents.items() if agent.present}
        record = MatchRecord(uuid.uuid4().hex[:12], self._rng.randrange(1 << 31), egos, "lobby")
        record.set_status("running")
        record.append({"type": "game_started", "lobby_id": lobby.lobby_id, "game_code": lobby.game_code,
                       "map": message.map, "players": list(message.players)})
        for agent in lobby.agents.values():
            agent.role = agent.special_role = agent.room = agent.last_action = None
            agent.alive = True
            agent.beliefs, agent.recalled, agent.memory = {}, [], {}
        lobby.match = record
        lobby.match_ids.append(record.match_id)
        lobby.state = "in_game"
        self._on_started(record)

    def _finish(self, lobby: Lobby, *, status: str, **values: Any) -> None:
        record = lobby.match
        assert record is not None
        record.set_status(status, **values)
        lobby.match = None
        lobby._decisions.clear()
        lobby._initiatives.clear()
        lobby.state = "lobby"
        self._on_finished(record)

    def _playing(self, lobby: Lobby, observation: Observation) -> LobbyAgent:
        if lobby.match is None:
            raise LobbyError("No game is running in this lobby.", 409)
        agent = self._agent(lobby, observation.agent_id)
        agent.role = observation.self_state.role.value
        agent.special_role = observation.self_state.special_role
        agent.alive = observation.self_state.alive
        agent.room = observation.self_state.room
        for event in observation.events:
            if event.kind in _DEATHS and event.target in lobby.agents:
                lobby.agents[event.target].alive = False
        return agent

    @staticmethod
    def _agent(lobby: Lobby, agent_id: str) -> LobbyAgent:
        agent = lobby.agents.get(agent_id)
        if agent is None:
            raise LobbyError(f"{agent_id!r} is not an agent of this lobby.")
        return agent

    def _payload(self, record: MatchRecord, observation: Observation, lobby: Lobby, **flags: bool) -> dict[str, Any]:
        agent = lobby.agents[observation.agent_id]
        choice, usable = _model_for(agent, observation)
        model = {"model_provider": choice.provider, "model": choice.model} if choice and usable else {}
        return {
            **model,
            "match_id": record.match_id,
            "match_seed": record.seed,
            "agent_id": observation.agent_id,
            "ego": record.egos[observation.agent_id],
            "players": sorted(record.egos),
            "source": lobby.source,
            "observation": to_dict(observation),
            **flags,
        }

    async def _observe(self, lobby: Lobby, agent: LobbyAgent, observation: Observation) -> None:
        record = lobby.match
        if record is None or observation.agent_id not in record.egos:
            return
        async with agent.lock:
            result = await self._runner.decide(
                agent_id=observation.agent_id,
                session_key=session_key(record.match_id, observation.agent_id),
                payload=self._payload(record, observation, lobby, offline=True, observe_only=True),
                mode=record.mode_of(observation.agent_id),
            )
        self._remember(agent, result)
        record.append({"type": "observation", "agent_id": observation.agent_id, "tick": observation.tick,
                       "perceived": [e.event_id for e in observation.events],
                       "memory": result.get("memory", {}), "error": result.get("error")})
        if result.get("initiative") and lobby.match is record:
            action = chosen_action(observation, {"action": result["initiative"]})
            if action.action is not ActionKind.WAIT:
                self.push_action(lobby, action, result.get("initiative_reason"))
        lobby.notify()

    async def _decide(self, lobby: Lobby, agent: LobbyAgent, request: DecisionRequest, received: float) -> None:
        record = lobby.match
        observation = request.observation
        if record is None or observation.agent_id not in record.egos:
            return
        async def turn(use_model: bool) -> dict[str, Any]:
            return await self._runner.decide(
                agent_id=observation.agent_id,
                session_key=session_key(record.match_id, observation.agent_id),
                payload=self._payload(record, observation, lobby, offline=not use_model),
                mode=record.mode_of(observation.agent_id),
            )

        # No cap, budget or timeout here: every agent thinks at once, for as long as its model needs. How long
        # the game waits for the answer is the server's setting (Nlght:DecisionTimeoutSeconds).
        async with agent.lock:
            if self._offline:
                result, mode = await turn(False), "policy:offline"
            elif not _model_for(agent, observation)[1]:
                # Its provider is not configured here: decide by policy, and say so in the counts.
                result, mode = await turn(False), "policy:no-model"
            else:
                result, mode = await turn(True), "model"
        self._remember(agent, result)
        action = chosen_action(observation, result)
        agent.decided[mode] += 1
        agent.last_action = to_dict(action)
        agent.last_latency_ms = int((time.monotonic() - received) * 1000)
        if lobby.match is not record:
            return
        lobby._decisions[request.request_id] = len(record.entries)
        record.append({**decision_entry(record, observation, result, mode, action),
                       "request_id": request.request_id, "accepted": None, "rejection": None,
                       "latency_ms": agent.last_latency_ms})
        self._send(lobby, ActionMessage(request.request_id, action))

    @staticmethod
    def _remember(agent: LobbyAgent, result: dict[str, Any]) -> None:
        if "beliefs" in result:
            agent.beliefs = result["beliefs"]
        if "recalled" in result:
            agent.recalled = list(result["recalled"])[:RECALLED_SHOWN]
        if "memory" in result:
            agent.memory = result["memory"]
        if "stance" in result:
            agent.timings, agent.stance = result.get("timings", {}), result["stance"]

    def _record_result(self, lobby: Lobby, result: ActionResult) -> None:
        agent = self._agent(lobby, result.agent_id)
        if result.request_id is None:
            # The verdict on an initiative; the server judges them in the order they were sent.
            agent.initiatives["accepted" if result.accepted else "refused"] += 1
            pending = lobby._initiatives.get(result.agent_id) or []
            index = pending.pop(0) if pending else None
        else:
            if not result.accepted:
                agent.refused += 1
            index = lobby._decisions.pop(result.request_id, None)
        if index is not None and lobby.match is not None:
            entry = lobby.match.entries[index]
            entry["accepted"], entry["rejection"] = result.accepted, result.reason

    def _send(self, lobby: Lobby, message: Any) -> None:
        lobby.outbox.put_nowait(encode(message))
        lobby.notify()

    def _spawn(self, coroutine: Any) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._done)
        return task

    def _done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("among_us.lobby.turn_failed | %s", task.exception(), exc_info=task.exception())

    async def drain(self) -> None:
        """Wait until every agent turn started so far has finished (tests)."""

        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
