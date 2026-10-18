"""Bridge protocol v1 between a game environment and the match orchestrator.

The same messages travel from the in-process simulation and from the C# game
bridge, so the agent architecture never depends on how the game is reached.
Parsing is strict: an unknown key is refused rather than ignored, because an
extra field in an observation — another player's role, say — is exactly the
leak the protocol exists to prevent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    """A message that does not match protocol v1."""


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


@dataclass(frozen=True)
class Hello:
    game_version: str
    source: str
    #: The agent-controlled players, in seat order.
    players: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchEnd:
    winner: str
    reason: str = ""


@dataclass(frozen=True)
class DecisionRequest:
    """Bridge → orchestrator: this agent is at a decision point."""

    observation: Observation


# -- encoding --------------------------------------------------------------

_TYPES: dict[type, str] = {
    Hello: "hello",
    DecisionRequest: "decision_request",
    SemanticAction: "action",
    ActionResult: "action_result",
    MatchEnd: "match_end",
}


def to_dict(value: Any) -> dict[str, Any]:
    """Plain JSON-ready dict of a protocol dataclass (enums become strings)."""

    return json.loads(json.dumps(asdict(value), default=str))


def encode(message: Any) -> str:
    """One protocol message as a single JSON line (NDJSON framing)."""

    kind = _TYPES.get(type(message))
    if kind is None:
        raise ProtocolError(f"not a protocol message: {type(message).__name__}")
    body = to_dict(message)
    return json.dumps({"type": kind, "protocol": PROTOCOL_VERSION, **body}, ensure_ascii=False)


def decode(line: str | bytes) -> Any:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"message is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("message must be a JSON object")
    body = dict(raw)
    kind = body.pop("type", None)
    protocol = body.pop("protocol", None)
    if protocol != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {protocol!r}")
    parsers = {
        "hello": lambda b: Hello(**{
            **(item := _exact(b, Hello)), "players": _strings(item.get("players"), "hello.players"),
        }),
        "decision_request": lambda b: DecisionRequest(
            observation=observation_from_dict(_exact(b, DecisionRequest)["observation"]),
        ),
        "action": action_from_dict,
        "action_result": lambda b: ActionResult(**_exact(b, ActionResult)),
        "match_end": lambda b: MatchEnd(**_exact(b, MatchEnd)),
    }
    parser = parsers.get(str(kind))
    if parser is None:
        raise ProtocolError(f"unknown message type: {kind!r}")
    return parser(body)


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
    )

