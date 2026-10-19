"""Protocol v2 between the nlght Impostor server and the orchestrator.

The server registers its bots per lobby (``LobbyRegistration`` →
``LobbyRegistered``), then each lobby has its own session: the server posts
game messages, the orchestrator streams actions back. The simulation hands the
same observations and actions over in process, so the agent architecture never
depends on how the game is reached.

Parsing is strict: an unknown key is refused rather than ignored, because an
extra field in an observation — another player's role, say — is exactly the
leak the protocol exists to prevent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

PROTOCOL_VERSION = 2


class ProtocolError(ValueError):
    """A message that does not match protocol v2."""


class Phase(StrEnum):
    TASKS = "TASKS"
    MEETING = "MEETING"
    VOTING = "VOTING"
    ENDED = "ENDED"


class Role(StrEnum):
    CREWMATE = "CREWMATE"
    IMPOSTOR = "IMPOSTOR"


class ActionKind(StrEnum):
    MOVE_TO = "MOVE_TO"
    DO_TASK = "DO_TASK"
    REPORT = "REPORT"
    KILL = "KILL"
    CALL_MEETING = "CALL_MEETING"
    SAY = "SAY"
    SABOTAGE = "SABOTAGE"
    VENT = "VENT"
    CLOSE_DOORS = "CLOSE_DOORS"
    #: Shapeshifter: take on another player's appearance for a while.
    SHAPESHIFT = "SHAPESHIFT"
    #: Phantom: turn invisible, and visible again.
    VANISH = "VANISH"
    APPEAR = "APPEAR"
    #: Keep a player in sight: escort someone trusted, shadow a suspect, or stalk a victim.
    FOLLOW = "FOLLOW"
    #: Stay where you are and watch who comes and goes.
    GUARD = "GUARD"
    VOTE = "VOTE"
    SKIP_VOTE = "SKIP_VOTE"
    WAIT = "WAIT"


class EventKind(StrEnum):
    """What an agent can perceive. Objective-only facts have no member here."""

    PLAYER_ENTERED = "PLAYER_ENTERED"
    PLAYER_LEFT = "PLAYER_LEFT"
    TASK_SEEN = "TASK_SEEN"
    KILL_WITNESSED = "KILL_WITNESSED"
    BODY_FOUND = "BODY_FOUND"
    MEETING_CALLED = "MEETING_CALLED"
    CHAT = "CHAT"
    VOTE_CAST = "VOTE_CAST"
    EJECTED = "EJECTED"
    NO_EJECTION = "NO_EJECTION"
    #: The impostor's own kill. Only the killer perceives it.
    KILL_COMMITTED = "KILL_COMMITTED"
    #: Someone was seen entering or leaving a vent; ``text`` says which.
    VENT_SEEN = "VENT_SEEN"
    #: A critical sabotage alarm started (``target`` is the system) or was fixed.
    SABOTAGE_ALARM = "SABOTAGE_ALARM"
    SABOTAGE_FIXED = "SABOTAGE_FIXED"
    #: The doors of ``room`` slammed shut; heard in the room and next to it.
    DOORS_CLOSED = "DOORS_CLOSED"
    #: Someone was seen changing shape: ``actor`` turned into ``target``.
    SHAPESHIFT_SEEN = "SHAPESHIFT_SEEN"
    #: Someone was seen vanishing into thin air, or appearing out of it.
    VANISH_SEEN = "VANISH_SEEN"
    APPEAR_SEEN = "APPEAR_SEEN"


@dataclass(frozen=True)
class VisiblePlayer:
    player_id: str
    room: str
    alive: bool = True


@dataclass(frozen=True)
class VisibleBody:
    body_id: str
    player_id: str
    room: str


@dataclass(frozen=True)
class ObservedEvent:
    """One event as the observing agent perceived it, never more."""

    event_id: str
    tick: int
    kind: EventKind
    room: str | None = None
    actor: str | None = None
    target: str | None = None
    text: str | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatMessage:
    """One line of the meeting chat, as everyone at the table sees it."""

    speaker: str
    text: str


@dataclass(frozen=True)
class LegalAction:
    action: ActionKind
    target: str | None = None


@dataclass(frozen=True)
class SelfState:
    player_id: str
    room: str
    role: Role
    alive: bool = True
    tasks_remaining: tuple[str, ...] = ()
    kill_ready: bool = False
    #: Impostors know their partners, as in the real game. Empty for crew.
    known_impostors: tuple[str, ...] = ()
    #: A special role on top of the team role ("SHAPESHIFTER", "PHANTOM"), only ever one's own.
    special_role: str | None = None
    #: Whose appearance a shapeshifter currently wears.
    disguised_as: str | None = None
    #: A phantom that cannot be seen right now.
    vanished: bool = False


@dataclass(frozen=True)
class Situation:
    """Where the game stands, as far as this player can know it: public facts and its own state."""

    #: The crew's task bar, 0..1; None when the game hides it (lobby setting, comms down).
    task_progress: float | None = None
    #: Emergency meetings this player can still call (crew).
    emergencies_left: int | None = None
    #: Ticks until this player's kill is ready (impostor).
    kill_cooldown: int | None = None
    #: Ticks since the last meeting ended, or since the game began.
    since_meeting: int | None = None


@dataclass(frozen=True)
class Observation:
    agent_id: str
    tick: int
    phase: Phase
    self_state: SelfState
    visible_players: tuple[VisiblePlayer, ...] = ()
    visible_bodies: tuple[VisibleBody, ...] = ()
    events: tuple[ObservedEvent, ...] = ()
    legal_actions: tuple[LegalAction, ...] = ()
    decision_reason: str = ""
    #: Everyone in the lobby, humans included — public knowledge in the real game.
    roster: tuple[str, ...] = ()
    #: The critical sabotage everyone can see right now ("reactor", "o2"), if any.
    alarm: str | None = None
    #: Non-critical sabotages in effect ("lights", "comms"); public, like the alarm.
    outages: tuple[str, ...] = ()
    #: In a meeting: its chat so far, word for word, as it stands on everyone's screen.
    meeting_chat: tuple[ChatMessage, ...] = ()
    situation: Situation = Situation()


@dataclass(frozen=True)
class SemanticAction:
    agent_id: str
    action: ActionKind
    target: str | None = None
    text: str | None = None
    goal: str | None = None
    evidence_ids: tuple[str, ...] = ()

    def legal_key(self) -> tuple[ActionKind, str | None]:
        return (self.action, self.target)


@dataclass(frozen=True)
class ActionResult:
    agent_id: str
    accepted: bool
    reason: str = ""
    #: The decision request the action answered; null for an action the orchestrator sent on its own.
    request_id: str | None = None


@dataclass(frozen=True)
class Hello:
    """In process only: the simulation introduces its players."""

    game_version: str
    source: str
    #: The agent-controlled players, in seat order.
    players: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchEnd:
    """In process only: the simulation reports its result."""

    winner: str
    reason: str = ""


# -- lobby registration (HTTP request and response bodies) -------------------


@dataclass(frozen=True)
class LobbyRegistration:
    """Server → orchestrator, when the host types ``!bots``: seat ``count`` agents in this lobby."""

    game_code: str
    game_version: str
    source: str
    count: int
    #: Names of everyone in the lobby now; an agent's name must not collide with one of them.
    players: tuple[str, ...] = ()


@dataclass(frozen=True)
class SeatedAgent:
    #: Always ``name.lower()``: the server keys every player by its lower-cased name.
    agent_id: str
    name: str
    ego: str


@dataclass(frozen=True)
class LobbyRegistered:
    """Orchestrator → server: the lobby's session and the agents to spawn, with exactly these names."""

    lobby_id: str
    token: str
    agents: tuple[SeatedAgent, ...] = ()


# -- lobby session messages --------------------------------------------------


@dataclass(frozen=True)
class GameStarted:
    """Server → orchestrator: a game began in the lobby; a new match starts."""

    map: str
    #: Everyone in the game, humans included.
    players: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservationUpdate:
    """Server → orchestrator: what an agent perceived since its last message. Nothing to answer."""

    observation: Observation


@dataclass(frozen=True)
class DecisionRequest:
    """Server → orchestrator: this agent is at a decision point and waits ``deadline_seconds``."""

    request_id: str
    deadline_seconds: float
    observation: Observation


@dataclass(frozen=True)
class ActionMessage:
    """Orchestrator → server: an action, answering ``request_id`` or (null) on the agent's own initiative."""

    request_id: str | None
    action: SemanticAction


@dataclass(frozen=True)
class AgentLeft:
    """Server → orchestrator: an agent left the lobby (``!bots off``, kicked, disconnected)."""

    agent_id: str
    reason: str = ""


@dataclass(frozen=True)
class GameEnded:
    """Server → orchestrator: the game is over. ``roles`` is the truth, for evaluation only."""

    winner: str
    reason: str = ""
    #: Every player's team role ("CREWMATE" / "IMPOSTOR"), humans included.
    roles: dict[str, str] = field(default_factory=dict)


# -- encoding --------------------------------------------------------------

_TYPES: dict[type, str] = {
    GameStarted: "game_started",
    ObservationUpdate: "observation",
    DecisionRequest: "decision_request",
    ActionMessage: "action",
    ActionResult: "action_result",
    AgentLeft: "agent_left",
    GameEnded: "game_ended",
}


def to_dict(value: Any) -> dict[str, Any]:
    """Plain JSON-ready dict of a protocol dataclass (enums become strings)."""

    return json.loads(json.dumps(asdict(value), default=str))


def encode(message: Any) -> str:
    """One lobby session message as a single JSON line."""

    kind = _TYPES.get(type(message))
    if kind is None:
        raise ProtocolError(f"not a protocol message: {type(message).__name__}")
    if isinstance(message, ActionMessage):
        # Flat on the wire: the action's fields next to the request it answers.
        body = {"request_id": message.request_id, **to_dict(message.action)}
    else:
        body = to_dict(message)
    return json.dumps({"type": kind, "protocol": PROTOCOL_VERSION, **body}, ensure_ascii=False)


def decode(line: str | bytes) -> Any:
    body = _versioned(line)
    kind = body.pop("type", None)
    parsers = {
        "game_started": lambda b: GameStarted(**{
            **(item := _exact(b, GameStarted)), "map": _str(item.get("map"), "game_started.map"),
            "players": _strings(item.get("players"), "game_started.players"),
        }),
        "observation": lambda b: ObservationUpdate(
            observation=observation_from_dict(_exact(b, ObservationUpdate).get("observation")),
        ),
        "decision_request": _decision_request,
        "action": _action_message,
        "action_result": _action_result,
        "agent_left": lambda b: AgentLeft(**{
            **(item := _exact(b, AgentLeft)), "agent_id": _str(item.get("agent_id"), "agent_left.agent_id"),
            "reason": str(item.get("reason", "")),
        }),
        "game_ended": _game_ended,
    }
    parser = parsers.get(str(kind))
    if parser is None:
        raise ProtocolError(f"unknown message type: {kind!r}")
    return parser(body)


def registration_from_dict(raw: Any) -> LobbyRegistration:
    body = _exact(_unversioned(raw), LobbyRegistration)
    count = body.get("count")
    if not isinstance(count, int) or isinstance(count, bool):
        raise ProtocolError("registration.count must be an integer")
    return LobbyRegistration(
        game_code=_str(body.get("game_code"), "registration.game_code"),
        game_version=_str(body.get("game_version"), "registration.game_version"),
        source=_str(body.get("source"), "registration.source"),
        count=count,
        players=_strings(body.get("players"), "registration.players"),
    )


def registered_to_dict(registered: LobbyRegistered) -> dict[str, Any]:
    return {"protocol": PROTOCOL_VERSION, **to_dict(registered)}


def _versioned(line: str | bytes) -> dict[str, Any]:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"message is not JSON: {exc}") from exc
    return _unversioned(raw)


def _unversioned(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ProtocolError("message must be a JSON object")
    body = dict(raw)
    protocol = body.pop("protocol", None)
    if protocol != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {protocol!r}")
    return body


def _decision_request(body: dict[str, Any]) -> DecisionRequest:
    item = _exact(body, DecisionRequest)
    deadline = item.get("deadline_seconds")
    if not isinstance(deadline, int | float) or isinstance(deadline, bool) or deadline <= 0:
        raise ProtocolError("decision_request.deadline_seconds must be a positive number")
    return DecisionRequest(
        request_id=_str(item.get("request_id"), "decision_request.request_id"),
        deadline_seconds=float(deadline),
        observation=observation_from_dict(item.get("observation")),
    )


def _action_message(body: dict[str, Any]) -> ActionMessage:
    body = dict(body)
    request_id = _optional_str(body.pop("request_id", None), "action.request_id")
    return ActionMessage(request_id, action_from_dict(body))


def _action_result(body: dict[str, Any]) -> ActionResult:
    item = _exact(body, ActionResult)
    if not isinstance(item.get("accepted"), bool):
        raise ProtocolError("action_result.accepted must be a boolean")
    return ActionResult(
        agent_id=_str(item.get("agent_id"), "action_result.agent_id"),
        accepted=item["accepted"],
        reason=str(item.get("reason", "")),
        request_id=_optional_str(item.get("request_id"), "action_result.request_id"),
    )


def _game_ended(body: dict[str, Any]) -> GameEnded:
    item = _exact(body, GameEnded)
    roles = item.get("roles") or {}
    if not isinstance(roles, dict) or not all(isinstance(k, str) for k in roles):
        raise ProtocolError("game_ended.roles must map players to roles")
    return GameEnded(
        winner=_str(item.get("winner"), "game_ended.winner"),
        reason=str(item.get("reason", "")),
        roles={player: _enum(Role, role, "game_ended.roles").value for player, role in roles.items()},
    )


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{where} must be a non-empty string")
    return value


def _exact(raw: Any, cls: type) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ProtocolError(f"{cls.__name__} must be an object")
    allowed = set(cls.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ProtocolError(f"{cls.__name__} has unknown fields: {sorted(unknown)}")
    return dict(raw)


def _enum(enum: type[StrEnum], value: Any, where: str) -> Any:
    try:
        return enum(value)
    except ValueError as exc:
        raise ProtocolError(f"{where}: invalid {enum.__name__} {value!r}") from exc


def _strings(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple) or not all(isinstance(v, str) for v in value):
        raise ProtocolError(f"{where} must be a list of strings")
    return tuple(value)


def _list(value: Any, where: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list | tuple):
        raise ProtocolError(f"{where} must be a list")
    return list(value)


def action_from_dict(raw: Any) -> SemanticAction:
    body = _exact(raw, SemanticAction)
    if not isinstance(body.get("agent_id"), str):
        raise ProtocolError("action.agent_id must be a string")
    return SemanticAction(
        agent_id=body["agent_id"],
        action=_enum(ActionKind, body.get("action"), "action.action"),
        target=_optional_str(body.get("target"), "action.target"),
        text=_optional_str(body.get("text"), "action.text"),
        goal=_optional_str(body.get("goal"), "action.goal"),
        evidence_ids=_strings(body.get("evidence_ids"), "action.evidence_ids"),
    )


def _optional_str(value: Any, where: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ProtocolError(f"{where} must be a string or null")


def observation_from_dict(raw: Any) -> Observation:
    body = _exact(raw, Observation)
    self_raw = _exact(body.get("self_state"), SelfState)
    self_state = SelfState(
        player_id=str(self_raw["player_id"]),
        room=str(self_raw["room"]),
        role=_enum(Role, self_raw.get("role"), "self_state.role"),
        alive=bool(self_raw.get("alive", True)),
        tasks_remaining=_strings(self_raw.get("tasks_remaining"), "self_state.tasks_remaining"),
        kill_ready=bool(self_raw.get("kill_ready", False)),
        known_impostors=_strings(self_raw.get("known_impostors"), "self_state.known_impostors"),
        special_role=_optional_str(self_raw.get("special_role"), "self_state.special_role"),
        disguised_as=_optional_str(self_raw.get("disguised_as"), "self_state.disguised_as"),
        vanished=bool(self_raw.get("vanished", False)),
    )
    events = tuple(
        ObservedEvent(**{
            **(item := _exact(e, ObservedEvent)),
            "kind": _enum(EventKind, item.get("kind"), "event.kind"),
            "evidence_ids": _strings(item.get("evidence_ids"), "event.evidence_ids"),
        })
        for e in _list(body.get("events"), "events")
    )
    return Observation(
        agent_id=str(body["agent_id"]),
        tick=int(body["tick"]),
        phase=_enum(Phase, body.get("phase"), "phase"),
        self_state=self_state,
        visible_players=tuple(
            VisiblePlayer(**_exact(p, VisiblePlayer)) for p in _list(body.get("visible_players"), "visible_players")
        ),
        visible_bodies=tuple(
            VisibleBody(**_exact(b, VisibleBody)) for b in _list(body.get("visible_bodies"), "visible_bodies")
        ),
        events=events,
        legal_actions=tuple(
            LegalAction(
                action=_enum(ActionKind, (item := _exact(a, LegalAction)).get("action"), "legal_action"),
                target=_optional_str(item.get("target"), "legal_action.target"),
            )
            for a in _list(body.get("legal_actions"), "legal_actions")
        ),
        decision_reason=str(body.get("decision_reason", "")),
        roster=_strings(body.get("roster"), "roster"),
        alarm=_optional_str(body.get("alarm"), "alarm"),
        outages=_strings(body.get("outages"), "outages"),
        meeting_chat=tuple(
            ChatMessage(
                speaker=_str((item := _exact(c, ChatMessage)).get("speaker"), "meeting_chat.speaker"),
                text=str(item.get("text", "")),
            )
            for c in _list(body.get("meeting_chat"), "meeting_chat")
        ),
        situation=_situation(body.get("situation")),
    )


def _situation(raw: Any) -> Situation:
    if raw is None:
        return Situation()
    body = _exact(raw, Situation)

    def number(key: str, kind: type) -> Any:
        value = body.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ProtocolError(f"situation.{key} must be a number or null")
        return kind(value)

    return Situation(
        task_progress=number("task_progress", float),
        emergencies_left=number("emergencies_left", int),
        kill_cooldown=number("kill_cooldown", int),
        since_meeting=number("since_meeting", int),
    )

