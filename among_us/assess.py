"""The agent's stance: where the game stands, whom it suspects and trusts, and what it means to do.

Deterministic and cheap, so a game turn can be decided fast: the weighing happens here, the model
only picks. In meetings the assessment also lines up what others said against what the agent
remembers — a contradiction becomes a conclusion of its own (``memory.judgement``), kept and
forgotten like any memory, so the stance the meeting shaped carries into the next game turns.

A tactic is held as a plan with a reason and the conditions that end it, so the agent does not
change course on a whim: following someone it trusts is a decision, not a reflex.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from among_us.ego import EgoProfile
from among_us.memory import judgement
from among_us.protocol import ActionKind, ChatMessage, EventKind, LegalAction, Observation, Phase, Role
from among_us.sim.map import ROOMS, SABOTAGE_ROOMS

_DEATHS = (EventKind.BODY_FOUND.value, EventKind.KILL_WITNESSED.value, EventKind.EJECTED.value)
#: Memories that place a player in a room.
_SIGHTINGS = (
    EventKind.PLAYER_ENTERED.value, EventKind.PLAYER_LEFT.value, EventKind.TASK_SEEN.value,
    EventKind.VENT_SEEN.value, EventKind.KILL_WITNESSED.value,
)
#: A sighting speaks against a claim only about the same moment: this many ticks before the body was found or
#: the meeting called. Earlier sightings are just where someone was earlier — people move.
_SAME_MOMENT = 4
_SUSPECT = 0.5


@dataclass(frozen=True)
class Contradiction:
    speaker: str
    player: str
    said: str
    seen: str
    tick: int
    memory_id: str

    def describe(self, now: int | None = None) -> str:
        whom = "they were" if self.player == self.speaker else f"{self.player} was"
        age = now - self.tick if now is not None else None
        ago = "" if age is None else " just now" if age <= 3 else " a little while ago" if age <= 15 else " earlier"
        return f"{self.speaker} says {whom} in {_room(self.said)}; you saw {self.player} in {_room(self.seen)}{ago}"


@dataclass
class Assessment:
    alive: list[str]
    dead: list[str]
    stage: str
    suspect: str | None
    trusted: list[str]
    contradictions: list[Contradiction] = field(default_factory=list)
    #: In a meeting: where the agent last saw each player before it, and when (tick).
    last_seen: dict[str, tuple[str, int]] = field(default_factory=dict)
    #: The move the plan calls for now, if one is legal.
    preferred: LegalAction | None = None
    #: The plan held across turns: tactic, action, target, reason, ends_if, since.
    plan: dict[str, Any] | None = None


def assess(
    observation: Observation,
    beliefs: dict[str, dict[str, Any]],
    recalled: Sequence[dict[str, Any]],
    ego: EgoProfile,
    plan: dict[str, Any] | None,
    known_dead: Iterable[str] = (),
) -> Assessment:
    me = observation.self_state
    meeting = observation.phase in (Phase.MEETING, Phase.VOTING)
    if meeting:
        # At the table it is certain: whoever is not sitting there is out of the game.
        alive = sorted({p.player_id for p in observation.visible_players} | {observation.agent_id})
        dead = sorted(set(observation.roster) - set(alive))
    else:
        dead = sorted(({m["details"].get("target") for m in recalled if m["kind"] in _DEATHS} | set(known_dead)) - {None})
        alive = sorted(set(observation.roster) - set(dead))
    suspects = [(b["suspicion"], p) for p, b in beliefs.items() if p in alive and b["suspicion"] >= _SUSPECT]
    suspect = max(suspects)[1] if suspects else None
    trusted = sorted((p for p, b in beliefs.items() if p in alive and b.get("trust", 0) >= ego.judgement.trust_at
                      and b["suspicion"] < _SUSPECT), key=lambda p: -beliefs[p]["trust"])
    progress = observation.situation.task_progress
    if len(alive) <= 4 or (progress is not None and progress >= 0.8):
        stage = "late"
    elif not dead and (progress is None or progress < 0.3):
        stage = "early"
    else:
        stage = "mid"
    result = Assessment(alive, dead, stage, suspect, trusted)
    if meeting:
        result.contradictions = contradictions(observation, recalled)
        result.last_seen = last_seen(observation, recalled)
        return result
    if observation.phase is Phase.TASKS and me.alive:
        _choose(observation, beliefs, ego, result, plan, recalled)
    return result


def conclusions(agent_id: str, found: Iterable[Contradiction]) -> list[dict[str, Any]]:
    """Each contradiction, as the conclusion the agent keeps: the speaker said something that does not fit."""

    return [
        judgement(agent_id, f"{c.speaker}:{c.memory_id}", c.tick, c.speaker, "+", [c.memory_id])
        for c in found
    ]


# -- what others said against what the agent saw ------------------------------------------------


def room_claims(text: str, speaker: str, players: Iterable[str], rooms: Iterable[str]) -> list[tuple[str, str]]:
    """Where a chat line puts whom: a room named in a sentence, about the player named in it, else the speaker."""

    names = sorted(set(players), key=len, reverse=True)
    vocabulary = sorted(set(rooms), key=len, reverse=True)
    claims = []
    for sentence in re.split(r"[.!?;]+", text.lower()):
        room = next((r for r in vocabulary if re.search(rf"\b{re.escape(_room(r))}\b", sentence)), None)
        if room is None:
            continue
        named = [p for p in names if p != speaker and re.search(rf"\b{re.escape(p)}\b", sentence)]
        claims += [(p, room) for p in named] if named else [(speaker, room)]
    return claims


#: Seen with one's own eyes and damning: what can justify an emergency meeting. Talk, votes and one's own
#: conclusions cannot — they are what meetings produce, and would call the next one.
_HARD_EVIDENCE = frozenset({"KILL_WITNESSED", "VENT_SEEN", "SHAPESHIFT_SEEN", "VANISH_SEEN", "APPEAR_SEEN"})


def fresh_evidence(observation: Observation, recalled: Sequence[dict[str, Any]], ids: Iterable[str]) -> list[str]:
    """Those of ``ids`` that are hard evidence the agent saw since the last meeting."""

    since = observation.situation.since_meeting
    start = observation.tick - since if since is not None else 0
    wanted = set(ids)
    return sorted(m["event_id"] for m in recalled
                  if m["event_id"] in wanted and m["kind"] in _HARD_EVIDENCE and int(m["tick"]) >= start)


def moment(observation: Observation, recalled: Sequence[dict[str, Any]]) -> int:
    """The moment a meeting is about: when the body was found, else when the meeting was called."""

    marks = [int(m["tick"]) for m in recalled if m["kind"] in ("BODY_FOUND", "MEETING_CALLED") and int(m["tick"]) <= observation.tick]
    return max(marks, default=observation.tick)


def last_seen(observation: Observation, recalled: Sequence[dict[str, Any]]) -> dict[str, tuple[str, int]]:
    seen: dict[str, tuple[str, int]] = {}
    for m in sorted(recalled, key=lambda m: int(m["tick"])):
        actor, room = m["details"].get("actor"), m["details"].get("room")
        if m["kind"] in _SIGHTINGS and actor and room and actor != observation.agent_id:
            seen[actor] = (room, int(m["tick"]))
    return seen


def contradictions(observation: Observation, recalled: Sequence[dict[str, Any]]) -> list[Contradiction]:
    """Claims in the chat that what the agent saw at the same moment speaks against — not merely an earlier place."""

    at = moment(observation, recalled)
    sightings = [m for m in recalled if m["kind"] in _SIGHTINGS and m["details"].get("room")
                 and at - _SAME_MOMENT <= int(m["tick"]) <= at]
    rooms = set(ROOMS) | {m["details"]["room"] for m in sightings}
    found: dict[tuple[str, str], Contradiction] = {}
    for message in observation.meeting_chat:
        if message.speaker == observation.agent_id:
            continue
        for player, room in room_claims(message.text, message.speaker, observation.roster, rooms):
            for memory in sightings:
                seen = memory["details"]["room"]
                if memory["details"].get("actor") == player and seen != room:
                    found.setdefault((message.speaker, memory["event_id"]), Contradiction(
                        message.speaker, player, room, seen, int(memory["tick"]), memory["event_id"]))
    return list(found.values())


def _room(room: str) -> str:
    return room.replace("_", " ")


# -- tactics ------------------------------------------------------------------------------------


def _choose(observation: Observation, beliefs: dict[str, dict[str, Any]], ego: EgoProfile, result: Assessment,
            plan: dict[str, Any] | None, recalled: Sequence[dict[str, Any]]) -> None:
    legal = set(observation.legal_actions)
    held = _still_valid(plan, observation, beliefs, legal)
    choose = _impostor if observation.self_state.role is Role.IMPOSTOR else _crew
    tactic = choose(observation, beliefs, ego, result, _kept(plan) if held else None, legal, recalled)
    if tactic is None:
        result.plan = None
        return
    name, action, reason, ends_if = tactic
    result.preferred = action
    since = plan["since"] if held and plan and plan.get("tactic") == name and plan.get("target") == action.target else observation.tick
    result.plan = {"tactic": name, "action": action.action.value, "target": action.target, "reason": reason,
                   "ends_if": ends_if, "since": since}


def _still_valid(plan: dict[str, Any] | None, observation: Observation, beliefs: dict[str, dict[str, Any]],
                 legal: set[LegalAction]) -> bool:
    if not plan:
        return False
    action = LegalAction(ActionKind(plan["action"]), plan.get("target"))
    if action not in legal:
        return False
    target = plan.get("target")
    suspicion = beliefs.get(target, {}).get("suspicion", 0.0) if target else 0.0
    if plan["tactic"] == "escort":
        return suspicion < _SUSPECT
    if plan["tactic"] == "shadow":
        return suspicion >= _SUSPECT - 0.1
    if plan["tactic"] == "stalk":
        return observation.self_state.kill_ready
    if plan["tactic"] == "agreed":
        # Agreed at the table: holds while the partner stays trustworthy.
        return action.action is not ActionKind.FOLLOW or suspicion < _SUSPECT
    return plan["tactic"] in ("watch", "blend") and bool(observation.visible_players)


Tactic = tuple[str, LegalAction, str, str]


def _kept(plan: dict[str, Any] | None) -> Tactic | None:
    """A plan still valid, as the tactic it was: it holds until its end condition, unless something urgent comes up."""

    if not plan:
        return None
    return plan["tactic"], LegalAction(ActionKind(plan["action"]), plan.get("target")), plan["reason"], plan["ends_if"]


def _crew(observation: Observation, beliefs: dict[str, dict[str, Any]], ego: EgoProfile, result: Assessment,
          kept: Tactic | None, legal: set[LegalAction], recalled: Sequence[dict[str, Any]]) -> Tactic | None:
    me = observation.self_state
    visible = {p.player_id for p in observation.visible_players}

    def can(kind: ActionKind, target: str | None = None) -> LegalAction | None:
        action = LegalAction(kind, target)
        return action if action in legal else None

    if reports := sorted(a.target for a in legal if a.action is ActionKind.REPORT and a.target):
        return "report", LegalAction(ActionKind.REPORT, reports[0]), "a body is here", "reported"
    for system in (observation.alarm, *observation.outages):
        if system and (fix := can(ActionKind.MOVE_TO, SABOTAGE_ROOMS.get(system))):
            return "fix", fix, f"{system} is sabotaged", "fixed"
    # An emergency meeting needs something new seen since the last one — never just the talk of the last meeting.
    if (result.suspect and beliefs[result.suspect]["suspicion"] >= ego.judgement.call_meeting_at
            and fresh_evidence(observation, recalled, beliefs[result.suspect]["evidence_ids"])
            and (call := can(ActionKind.CALL_MEETING))):
        return "accuse", call, f"you saw something about {result.suspect} since the last meeting", "the meeting starts"
    if kept and kept[0] in ("agreed", "escort", "shadow", "watch"):
        return kept
    escort = next((p for p in result.trusted if p in visible and can(ActionKind.FOLLOW, p)), None)
    lonely = result.stage == "late" and ego.judgement.group_affinity >= 0.5
    if escort and lonely:
        return ("escort", LegalAction(ActionKind.FOLLOW, escort),
                f"few are left and you trust {escort}", f"{escort} looks suspicious or dies")
    if tasks := sorted(a.target for a in legal if a.action is ActionKind.DO_TASK and a.target):
        return "tasks", LegalAction(ActionKind.DO_TASK, tasks[0]), "your tasks win the game", "the task is done"
    if me.tasks_remaining:
        return None   # on the way to the next task: the policy knows the rooms of the simulated ship
    if result.suspect in visible and ego.judgement.boldness >= 0.5 and (shadow := can(ActionKind.FOLLOW, result.suspect)):
        return ("shadow", shadow, f"your tasks are done and {result.suspect} is your suspect",
                f"{result.suspect} no longer looks suspicious, or you lose them")
    if escort and (result.stage == "late" or ego.judgement.group_affinity >= 0.5):
        return ("escort", LegalAction(ActionKind.FOLLOW, escort), f"your tasks are done and you trust {escort}",
                f"{escort} looks suspicious or dies")
    if visible and ego.judgement.group_affinity >= 0.5 and (watch := can(ActionKind.GUARD)):
        return "watch", watch, "your tasks are done; you stay with others and watch", "you are alone"
    return None


def _impostor(observation: Observation, beliefs: dict[str, dict[str, Any]], ego: EgoProfile, result: Assessment,
              kept: Tactic | None, legal: set[LegalAction], recalled: Sequence[dict[str, Any]]) -> Tactic | None:
    me = observation.self_state
    crew = [p.player_id for p in observation.visible_players if p.player_id not in me.known_impostors]

    def first(kind: ActionKind) -> list[str]:
        return sorted(a.target for a in legal if a.action is kind and a.target)

    if any(e.kind is EventKind.KILL_COMMITTED for e in observation.events):
        if vents := first(ActionKind.VENT):
            return "escape", LegalAction(ActionKind.VENT, vents[0]), "leave the body unseen", "you are away"
        if rooms := [r for r in first(ActionKind.MOVE_TO) if r != me.room]:
            return "escape", LegalAction(ActionKind.MOVE_TO, rooms[0]), "leave the body", "you are away"
    kills = first(ActionKind.KILL)
    if me.kill_ready and len(crew) == 1 and kills:
        return "strike", LegalAction(ActionKind.KILL, kills[0]), f"you are alone with {kills[0]}", "done"
    if kept and kept[0] in ("agreed", "stalk", "blend"):
        return kept
    if me.kill_ready and crew and len(crew) <= 2 and ego.judgement.boldness >= 0.4:
        victim = sorted(crew, key=lambda p: beliefs.get(p, {}).get("trust", 0.0))[0]
        if LegalAction(ActionKind.FOLLOW, victim) in legal:
            return "stalk", LegalAction(ActionKind.FOLLOW, victim), f"stay near {victim} until nobody else watches", "you lose them"
    cooldown = observation.situation.kill_cooldown
    if cooldown is not None and cooldown <= 2 and ego.judgement.boldness >= 0.6 and "lights" in first(ActionKind.SABOTAGE):
        return "isolate", LegalAction(ActionKind.SABOTAGE, "lights"), "darkness splits the crew before you strike", "the lights are out"
    if tasks := first(ActionKind.DO_TASK):
        why = "fake a task where others can see you" if crew else "fake a task to keep your cover"
        return "alibi", LegalAction(ActionKind.DO_TASK, tasks[0]), why, "the task is 'done'"
    if crew and ego.judgement.group_affinity >= 0.5 and LegalAction(ActionKind.GUARD) in legal:
        return "blend", LegalAction(ActionKind.GUARD), "stay with the crew and look busy", "they leave"
    return None


def chat_lines(chat: Sequence[ChatMessage], self_id: str) -> list[str]:
    return [f"{'you' if m.speaker == self_id else m.speaker}: {m.text}" for m in chat]
