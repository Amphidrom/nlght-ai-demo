"""A seed-deterministic, tick-based Among Us simulation.

It plays the role the nlght Impostor server plays for the real game: it owns the
truth, filters what each player perceives, and accepts only legal actions.
Agents never read the engine; they read the observations it hands out.

Perception follows appearance. A shapeshifter is perceived as whoever it looks
like, a vanished phantom is not perceived at all, and with the lights out a
crewmate sees that someone is there but not who.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from among_us.domain import GameEvent
from among_us.protocol import (
    ChatMessage,
    ActionKind,
    ActionResult,
    EventKind,
    LegalAction,
    MatchEnd,
    ObservedEvent,
    Observation,
    Phase,
    Role,
    SelfState,
    SemanticAction,
    VisibleBody,
    VisiblePlayer,
)
from among_us.sim.map import (
    ADJACENCY,
    CRITICAL_SABOTAGES,
    DOOR_ROOMS,
    EMERGENCY_BUTTON_ROOM,
    ROOMS,
    SABOTAGE_ROOMS,
    SPAWN_ROOM,
    TASKS,
    next_hop,
    vent_reachable,
)

_SALIENT = frozenset({
    EventKind.KILL_WITNESSED, EventKind.BODY_FOUND, EventKind.MEETING_CALLED,
    EventKind.KILL_COMMITTED, EventKind.VENT_SEEN, EventKind.SABOTAGE_ALARM,
    EventKind.DOORS_CLOSED, EventKind.SHAPESHIFT_SEEN, EventKind.VANISH_SEEN,
})

MIN_PLAYERS = 4
MAX_PLAYERS = 15

SHAPESHIFTER = "SHAPESHIFTER"
PHANTOM = "PHANTOM"
SPECIAL_ROLES = (SHAPESHIFTER, PHANTOM)

#: An observer sees someone, but cannot tell who (lights out).
SOMEONE = "someone"
#: Marks a player nobody else can perceive (a vanished phantom).
_HIDDEN = object()


def max_impostors(players: int) -> int:
    """The game's own lobby limit: 1 up to six players, 2 up to eight, then 3."""
    return 1 if players <= 6 else 2 if players <= 8 else 3


@dataclass
class _Player:
    player_id: str
    role: Role
    tasks: list[str]
    special_role: str | None = None
    room: str = SPAWN_ROOM
    alive: bool = True
    intent: SemanticAction | None = None
    button_used: bool = False
    kill_cooldown: int = 0
    pending: list[ObservedEvent] = field(default_factory=list)
    seen_bodies: set[str] = field(default_factory=set)
    phase_seen: Phase | None = None
    task_progress: int = 0
    #: Kill targets present at the last observation. A kill opportunity is a
    #: decision point only when this changes, not on every tick it persists.
    kill_targets_seen: frozenset[str] = frozenset()
    #: Shapeshifter: whose appearance it wears, and for how long.
    disguise: str | None = None
    disguise_ticks: int = 0
    #: Phantom: invisible, and for how long.
    vanished: bool = False
    vanish_ticks: int = 0
    #: Cooldown of the special role's ability.
    ability_cooldown: int = 0


@dataclass(frozen=True)
class _Body:
    body_id: str
    player_id: str
    room: str


class SimEngine:
    """One match. Every random choice comes from ``seed``."""

    source = "nlght-ai-demo/among-us-sim"
    game_version = "sim-1"

    def __init__(
        self,
        players: Sequence[str],
        *,
        seed: int,
        impostors: int = 1,
        impostor_roles: Sequence[str] = (),
        tasks_per_player: int = 4,
        task_ticks: int = 3,
        kill_cooldown: int = 5,
        sabotage_cooldown: int = 12,
        sabotage_countdown: int = 10,
        door_ticks: int = 5,
        door_cooldown: int = 15,
        shapeshift_ticks: int = 8,
        vanish_ticks: int = 6,
        ability_cooldown: int = 10,
        meeting_rounds: int = 1,
        max_ticks: int = 240,
    ) -> None:
        if len(set(players)) != len(players) or not MIN_PLAYERS <= len(players) <= MAX_PLAYERS:
            raise ValueError(f"a match needs {MIN_PLAYERS} to {MAX_PLAYERS} distinct players")
        if not 1 <= impostors <= max_impostors(len(players)):
            raise ValueError(f"{len(players)} players allow 1 to {max_impostors(len(players))} impostors")
        unknown = set(impostor_roles) - set(SPECIAL_ROLES)
        if unknown or len(impostor_roles) > impostors:
            raise ValueError(f"impostor_roles must name at most {impostors} of {SPECIAL_ROLES}")
        self.seed = seed
        self._rng = random.Random(seed)
        order = sorted(players)
        impostor_ids = set(self._rng.sample(order, impostors))
        roles_left = list(impostor_roles)
        self._players: dict[str, _Player] = {}
        for pid in order:
            role = Role.IMPOSTOR if pid in impostor_ids else Role.CREWMATE
            # Impostors are dealt a task list too: it is their cover story.
            tasks = self._rng.sample(sorted(TASKS), tasks_per_player)
            special = roles_left.pop(0) if role is Role.IMPOSTOR and roles_left else None
            self._players[pid] = _Player(pid, role, tasks, special)
        self._kill_cooldown = kill_cooldown
        self._task_ticks = task_ticks
        self._sabotage_cooldown_ticks = sabotage_cooldown
        self._sabotage_countdown_ticks = sabotage_countdown
        self._door_ticks = door_ticks
        self._door_cooldown_ticks = door_cooldown
        self._shapeshift_ticks = shapeshift_ticks
        self._vanish_ticks = vanish_ticks
        self._ability_cooldown_ticks = ability_cooldown
        self._meeting_rounds = meeting_rounds
        self._meeting_chat: list[ChatMessage] = []
        self._max_ticks = max_ticks
        #: Shared by the impostor team, as in the real game.
        self._sabotage_cooldown = sabotage_cooldown
        self._door_cooldown = 0
        #: Active sabotage: (system, ticks left). Only critical ones count down.
        self._sabotage: tuple[str, int] | None = None
        #: Rooms with shut doors → ticks until they open.
        self._doors: dict[str, int] = {}
        self.tick = 0
        self.phase = Phase.TASKS
        self.winner: MatchEnd | None = None
        self.events: list[GameEvent] = []
        #: tick → where every living player stood after that tick's moves.
        #: Ground truth for checking location claims; never observed.
        self.positions: dict[int, dict[str, str]] = {}
        self._bodies: dict[str, _Body] = {}
        self._statements: dict[str, SemanticAction] = {}
        self._votes: dict[str, str | None] = {}
        self._round = 0
        for player in self._players.values():
            if player.role is Role.IMPOSTOR:
                player.kill_cooldown = kill_cooldown
        self._record_positions()

    # -- read side -----------------------------------------------------------

    @property
    def player_ids(self) -> tuple[str, ...]:
        return tuple(self._players)

    def alive(self) -> tuple[str, ...]:
        return tuple(pid for pid, p in self._players.items() if p.alive)

    def spectator_view(self) -> dict:
        """The whole truth, for the human spectator only. Never an agent input."""
        return {
            "tick": self.tick,
            "phase": self.phase.value,
            "players": [
                {"player_id": pid, "room": p.room, "alive": p.alive, "role": p.role.value,
                 "special_role": p.special_role, "disguised_as": p.disguise, "vanished": p.vanished,
                 "tasks_remaining": len(p.tasks)}
                for pid, p in self._players.items()
            ],
            "bodies": [{"player_id": b.player_id, "room": b.room} for b in self._bodies.values()],
            "sabotage": {"system": self._sabotage[0], "ticks_left": self._sabotage[1] if self._sabotage[0] in CRITICAL_SABOTAGES else None}
            if self._sabotage else None,
            "doors_closed": sorted(self._doors),
            "winner": self.winner.winner if self.winner else None,
        }

    def role_of(self, pid: str) -> Role:
        """Ground truth for evaluation. Never part of an observation."""
        return self._players[pid].role

    def special_role_of(self, pid: str) -> str | None:
        return self._players[pid].special_role

    def needs_decision(self, pid: str) -> bool:
        player = self._players[pid]
        if not player.alive or self.phase is Phase.ENDED:
            return False
        if self.phase is Phase.MEETING:
            return pid not in self._statements
        if self.phase is Phase.VOTING:
            return pid not in self._votes
        if player.intent is None or player.phase_seen is not Phase.TASKS:
            return True
        if any(event.kind in _SALIENT for event in player.pending):
            return True
        targets = frozenset(self._kill_targets(pid))
        return (
            player.role is Role.IMPOSTOR
            and player.kill_cooldown == 0
            and player.intent.action is not ActionKind.KILL
            and bool(targets)
            and targets != player.kill_targets_seen
        )

    def observe(self, pid: str) -> Observation:
        player = self._players[pid]
        events, player.pending = tuple(player.pending), []
        reason = self._decision_reason(player, events)
        player.phase_seen = self.phase
        player.kill_targets_seen = frozenset(self._kill_targets(pid)) if player.kill_cooldown == 0 else frozenset()
        in_meeting = self.phase in (Phase.MEETING, Phase.VOTING)
        if in_meeting:
            # At the meeting table everyone is visible, but not where they were: that is what the
            # discussion is about. The real game (and the nlght Impostor server) reports it the same way.
            visible = tuple(
                VisiblePlayer(other.player_id, "meeting", other.alive)
                for other in self._players.values()
                if other.player_id != pid and other.alive
            )
        else:
            visible = tuple(
                VisiblePlayer(seen, other.room, other.alive)
                for other in self._players.values()
                if other.player_id != pid and other.alive and other.room == player.room
                and (seen := self._apparent(other.player_id, pid)) is not _HIDDEN
            )
        bodies = () if in_meeting else tuple(
            VisibleBody(b.body_id, b.player_id, b.room) for b in self._bodies.values() if b.room == player.room
        )
        crew_blind_to_tasks = self._outage("comms") and player.role is Role.CREWMATE
        return Observation(
            agent_id=pid,
            tick=self.tick,
            phase=self.phase,
            self_state=SelfState(
                player_id=pid,
                room=player.room,
                role=player.role,
                alive=player.alive,
                tasks_remaining=() if crew_blind_to_tasks else tuple(player.tasks),
                kill_ready=player.role is Role.IMPOSTOR and player.kill_cooldown == 0 and not player.vanished,
                known_impostors=tuple(
                    other for other, p in self._players.items()
                    if p.role is Role.IMPOSTOR and other != pid
                ) if player.role is Role.IMPOSTOR else (),
                special_role=player.special_role,
                disguised_as=player.disguise,
                vanished=player.vanished,
            ),
            visible_players=visible,
            visible_bodies=bodies,
            events=events,
            legal_actions=self.legal_actions(pid),
            decision_reason=reason,
            roster=tuple(sorted(self._players)),
            alarm=self._sabotage[0] if self._sabotage and self._sabotage[0] in CRITICAL_SABOTAGES else None,
            outages=tuple(sorted({self._sabotage[0]} - CRITICAL_SABOTAGES)) if self._sabotage else (),
            meeting_chat=tuple(self._meeting_chat) if in_meeting else (),
        )

    def legal_actions(self, pid: str) -> tuple[LegalAction, ...]:
        player = self._players[pid]
        if not player.alive or self.phase is Phase.ENDED:
            return ()
        if self.phase is Phase.MEETING:
            return (LegalAction(ActionKind.SAY), LegalAction(ActionKind.WAIT))
        if self.phase is Phase.VOTING:
            return tuple(
                LegalAction(ActionKind.VOTE, other) for other in self.alive() if other != pid
            ) + (LegalAction(ActionKind.SKIP_VOTE),)
        impostor = player.role is Role.IMPOSTOR
        actions = [LegalAction(ActionKind.MOVE_TO, room) for room in ROOMS if room != player.room]
        if impostor or not self._outage("comms"):
            # With comms down a crewmate cannot see its task list.
            actions += [LegalAction(ActionKind.DO_TASK, t) for t in player.tasks if TASKS[t] == player.room]
        actions += [
            LegalAction(ActionKind.REPORT, b.body_id) for b in self._bodies.values() if b.room == player.room
        ]
        if impostor:
            if player.kill_cooldown == 0 and not player.vanished:
                actions += [LegalAction(ActionKind.KILL, target) for target in self._kill_targets(pid)]
            actions += [LegalAction(ActionKind.VENT, room) for room in vent_reachable(player.room)]
            if self._sabotage is None and self._sabotage_cooldown == 0:
                actions += [LegalAction(ActionKind.SABOTAGE, system) for system in sorted(SABOTAGE_ROOMS)]
            if self._door_cooldown == 0:
                actions += [LegalAction(ActionKind.CLOSE_DOORS, room) for room in sorted(DOOR_ROOMS - set(self._doors))]
            actions += self._ability_actions(player)
        if player.room == EMERGENCY_BUTTON_ROOM and not player.button_used and not self._critical():
            actions.append(LegalAction(ActionKind.CALL_MEETING))
        actions.append(LegalAction(ActionKind.WAIT))
        return tuple(actions)

    def _ability_actions(self, player: _Player) -> list[LegalAction]:
        if player.special_role == SHAPESHIFTER and player.disguise is None and player.ability_cooldown == 0:
            return [LegalAction(ActionKind.SHAPESHIFT, other) for other in self.alive() if other != player.player_id]
        if player.special_role == PHANTOM:
            if player.vanished:
                return [LegalAction(ActionKind.APPEAR)]
            if player.ability_cooldown == 0:
                return [LegalAction(ActionKind.VANISH)]
        return []

    # -- write side ----------------------------------------------------------

    def submit(self, action: SemanticAction) -> ActionResult:
        player = self._players.get(action.agent_id)
        if player is None or not player.alive:
            return ActionResult(action.agent_id, False, "not an active player")
        legal = {(a.action, a.target) for a in self.legal_actions(action.agent_id)}
        if action.legal_key() not in legal:
            return ActionResult(action.agent_id, False, f"illegal in {self.phase}: {action.action} {action.target}")
        if self.phase is Phase.MEETING:
            self._statements[player.player_id] = action
            if action.action is ActionKind.SAY and action.text:
                self._meeting_chat.append(ChatMessage(player.player_id, action.text))
                self._emit(
                    "CHAT", EventKind.CHAT, self.alive(), room=None, actor=player.player_id,
                    text=action.text, evidence_ids=action.evidence_ids, exclude_actor=False,
                )
            return ActionResult(action.agent_id, True)
        if self.phase is Phase.VOTING:
            self._votes[player.player_id] = action.target if action.action is ActionKind.VOTE else None
            return ActionResult(action.agent_id, True)
        if player.intent is None or player.intent.legal_key() != action.legal_key():
            player.task_progress = 0
        player.intent = action
        return ActionResult(action.agent_id, True)

    def advance(self) -> MatchEnd | None:
        if self.phase is Phase.ENDED:
            return self.winner
        if self.phase is Phase.MEETING:
            if set(self.alive()) <= set(self._statements):
                self._statements = {}
                self._round += 1
                if self._round >= self._meeting_rounds:
                    self.phase = Phase.VOTING
            return None
        if self.phase is Phase.VOTING:
            if set(self.alive()) <= set(self._votes):
                self._resolve_votes()
            return self._check_end()
        self.tick += 1
        # Kills, reports, the button, sabotage, doors, vents and abilities are instantaneous:
        # they resolve before anyone walks, so a victim cannot step away from a kill already made.
        instant = (
            ActionKind.KILL, ActionKind.REPORT, ActionKind.CALL_MEETING, ActionKind.SABOTAGE, ActionKind.VENT,
            ActionKind.CLOSE_DOORS, ActionKind.SHAPESHIFT, ActionKind.VANISH, ActionKind.APPEAR,
        )
        for first_pass in (True, False):
            for pid in sorted(self.alive()):
                player = self._players[pid]
                is_instant = player.intent is not None and player.intent.action in instant
                if player.alive and self.phase is Phase.TASKS and is_instant == first_pass:
                    self._execute(player)
        if self.phase is Phase.TASKS:
            self._tick_timers()
        self._sabotage_cooldown = max(0, self._sabotage_cooldown - 1)
        self._door_cooldown = max(0, self._door_cooldown - 1)
        if self.phase is Phase.TASKS:
            outcome = self._tick_sabotage()
            if outcome is not None:
                return outcome
        self._record_positions()
        if self.tick >= self._max_ticks and self.phase is Phase.TASKS:
            return self._end("draw", "tick limit reached")
        return self._check_end()

    # -- internals -------------------------------------------------------------

    def _record_positions(self) -> None:
        self.positions[self.tick] = {pid: p.room for pid, p in self._players.items() if p.alive}

    def _critical(self) -> bool:
        return self._sabotage is not None and self._sabotage[0] in CRITICAL_SABOTAGES

    def _outage(self, system: str) -> bool:
        return self._sabotage is not None and self._sabotage[0] == system

    def _apparent(self, pid: str, observer: str) -> object:
        """Who ``observer`` perceives ``pid`` as: an id, ``SOMEONE``, or ``_HIDDEN``."""
        if pid == observer:
            return pid
        player = self._players[pid]
        if player.vanished:
            return _HIDDEN
        if self._outage("lights") and self._players[observer].role is Role.CREWMATE:
            return SOMEONE
        return player.disguise or pid

    def _tick_timers(self) -> None:
        for player in self._players.values():
            player.kill_cooldown = max(0, player.kill_cooldown - 1)
            player.ability_cooldown = max(0, player.ability_cooldown - 1)
            if player.disguise is not None:
                player.disguise_ticks -= 1
                if player.disguise_ticks <= 0:
                    self._revert(player)
            if player.vanished:
                player.vanish_ticks -= 1
                if player.vanish_ticks <= 0:
                    self._appear(player)
        for room in list(self._doors):
            self._doors[room] -= 1
            if self._doors[room] <= 0:
                del self._doors[room]

    def _execute(self, player: _Player) -> None:
        intent = player.intent
        if intent is None:
            return
        kind = intent.action
        if kind is ActionKind.MOVE_TO and intent.target:
            hop = next_hop(player.room, intent.target)
            if hop is None:
                player.intent = None
                return
            if player.room in self._doors or hop in self._doors:
                return   # the doors are shut: wait, the intent stays
            origin = player.room
            others_from = self._others_in(origin, player.player_id)
            others_to = self._others_in(hop, player.player_id)
            self._emit("MOVED", EventKind.PLAYER_LEFT, others_from, room=origin,
                       actor=player.player_id, target=hop, text=None)
            player.room = hop
            if others_to:
                self._emit("MOVED", EventKind.PLAYER_ENTERED, others_to, room=hop,
                           actor=player.player_id, target=origin, text=None)
            for other in others_to:
                self._emit("SIGHTING", EventKind.PLAYER_ENTERED, [player.player_id], room=hop,
                           actor=other, text="already here")
            self._discover_bodies(player)
            if hop == intent.target:
                player.intent = None
            return
        if kind is ActionKind.DO_TASK and intent.target in player.tasks:
            # A task takes several ticks; it is seen by whoever is present when it starts.
            if player.task_progress == 0:
                objective = "TASK_STARTED" if player.role is Role.CREWMATE else "FAKE_TASK"
                self._emit(objective, EventKind.TASK_SEEN, self._others_in(player.room, player.player_id),
                           room=player.room, actor=player.player_id, target=intent.target)
            player.task_progress += 1
            if player.task_progress < self._task_ticks:
                return
            player.task_progress = 0
            player.tasks.remove(intent.target)
        elif kind is ActionKind.KILL and intent.target:
            victim = self._players.get(intent.target)
            if victim and victim.alive and victim.room == player.room and player.kill_cooldown == 0 and not player.vanished:
                victim.alive = False
                victim.pending = []
                victim.intent = None
                witnesses = [p for p in self._others_in(player.room, player.player_id) if p != victim.player_id]
                event = self._emit("KILL", EventKind.KILL_WITNESSED, witnesses, room=player.room,
                                   actor=player.player_id, target=victim.player_id, target_is_player=True)
                # The killer remembers the deed itself, as a different kind of perception.
                own = ObservedEvent(event.event_id, self.tick, EventKind.KILL_COMMITTED, player.room,
                                    player.player_id, victim.player_id)
                event.views[player.player_id] = own
                player.pending.append(own)
                body_id = f"body_{victim.player_id}"
                self._bodies[body_id] = _Body(body_id, victim.player_id, player.room)
                player.seen_bodies.add(body_id)
                player.kill_cooldown = self._kill_cooldown
                for pid in witnesses:
                    self._players[pid].seen_bodies.add(body_id)
        elif kind is ActionKind.VENT and intent.target in vent_reachable(player.room):
            origin = player.room
            self._emit("VENT", EventKind.VENT_SEEN, self._others_in(origin, player.player_id), room=origin,
                       actor=player.player_id, text="entered")
            player.room = intent.target
            self._emit("VENT", EventKind.VENT_SEEN, self._others_in(player.room, player.player_id),
                       room=player.room, actor=player.player_id, text="exited")
            self._discover_bodies(player)
        elif kind is ActionKind.SABOTAGE and intent.target in SABOTAGE_ROOMS and self._sabotage is None:
            self._sabotage = (intent.target, self._sabotage_countdown_ticks)
            self._sabotage_cooldown = self._sabotage_cooldown_ticks
            # The alarm sounds for everyone, the saboteur included; nobody perceives who pulled it.
            self._emit("SABOTAGE", EventKind.SABOTAGE_ALARM, self.alive(), room=None,
                       target=intent.target, exclude_actor=False)
        elif kind is ActionKind.CLOSE_DOORS and intent.target in DOOR_ROOMS and intent.target not in self._doors:
            room = intent.target
            self._doors[room] = self._door_ticks
            self._door_cooldown = self._door_cooldown_ticks
            # Heard in the room and next to it; nobody perceives who shut them.
            hearers = [pid for pid in self.alive() if self._players[pid].room == room or self._players[pid].room in ADJACENCY[room]]
            self._emit("DOORS", EventKind.DOORS_CLOSED, hearers, room=room, exclude_actor=False)
        elif kind is ActionKind.SHAPESHIFT and intent.target and player.special_role == SHAPESHIFTER and player.disguise is None:
            witnesses = self._others_in(player.room, player.player_id)
            self._emit_change("SHAPESHIFT", EventKind.SHAPESHIFT_SEEN, player, witnesses, before=player.player_id,
                              after=intent.target, text="shifted")
            player.disguise = intent.target
            player.disguise_ticks = self._shapeshift_ticks
            player.ability_cooldown = self._ability_cooldown_ticks
        elif kind is ActionKind.VANISH and player.special_role == PHANTOM and not player.vanished:
            self._emit_change("VANISH", EventKind.VANISH_SEEN, player, self._others_in(player.room, player.player_id),
                              before=player.player_id, after=None, text="vanished")
            player.vanished = True
            player.vanish_ticks = self._vanish_ticks
        elif kind is ActionKind.APPEAR and player.vanished:
            self._appear(player)
        elif kind is ActionKind.REPORT and intent.target in self._bodies:
            body = self._bodies[intent.target]
            if body.room == player.room:
                self._start_meeting(player, body.player_id, "body reported")
        elif kind is ActionKind.CALL_MEETING and player.room == EMERGENCY_BUTTON_ROOM and not player.button_used:
            player.button_used = True
            self._start_meeting(player, None, "emergency meeting")
        player.intent = None

    def _revert(self, player: _Player) -> None:
        """The shape wears off — and whoever watches sees who it really was."""
        self._emit_change("REVERT", EventKind.SHAPESHIFT_SEEN, player, self._others_in(player.room, player.player_id),
                          before=player.disguise, after=player.player_id, text="reverted")
        player.disguise = None
        player.disguise_ticks = 0

    def _appear(self, player: _Player) -> None:
        player.vanished = False
        player.vanish_ticks = 0
        player.ability_cooldown = self._ability_cooldown_ticks
        self._emit("APPEAR", EventKind.APPEAR_SEEN, self._others_in(player.room, player.player_id),
                   room=player.room, actor=player.player_id, text="appeared")

    def _discover_bodies(self, player: _Player) -> None:
        for body in self._bodies.values():
            if body.room == player.room and body.body_id not in player.seen_bodies:
                player.seen_bodies.add(body.body_id)
                self._emit("BODY_DISCOVERED", EventKind.BODY_FOUND, [player.player_id],
                           room=body.room, actor=None, target=body.player_id)

    def _start_meeting(self, caller: _Player, victim: str | None, reason: str) -> None:
        # At the table every disguise drops and every phantom is back: identities are real.
        for p in self._players.values():
            p.disguise, p.disguise_ticks = None, 0
            p.vanished, p.vanish_ticks = False, 0
        self._emit("MEETING", EventKind.MEETING_CALLED, self.alive(), room=caller.room,
                   actor=caller.player_id, target=victim, text=reason, exclude_actor=False)
        self._bodies.clear()
        self._doors.clear()
        if self._sabotage is not None:
            self._sabotage = None
            self._emit("SABOTAGE_RESOLVED", EventKind.SABOTAGE_FIXED, self.alive(), room=None, text="meeting")
        self.phase = Phase.MEETING
        self._round = 0
        self._meeting_chat = []
        self._statements = {}
        self._votes = {}
        for p in self._players.values():
            p.intent = None

    def _resolve_votes(self) -> None:
        tally = Counter(target for target in self._votes.values())
        for voter, target in sorted(self._votes.items()):
            self._emit("VOTE", EventKind.VOTE_CAST, self.alive(), room=None, actor=voter,
                       target=target, exclude_actor=False, target_is_player=True)
        skip = tally.pop(None, 0)
        ranked = tally.most_common()
        ejected = None
        if ranked and ranked[0][1] > skip and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            ejected = ranked[0][0]
        if ejected:
            self._emit("EJECT", EventKind.EJECTED, self.alive(), room=None, target=ejected, exclude_actor=False,
                       target_is_player=True)
            self._players[ejected].alive = False
            self._players[ejected].pending = []
        else:
            self._emit("NO_EJECT", EventKind.NO_EJECTION, self.alive(), room=None, exclude_actor=False)
        self._votes = {}
        self.phase = Phase.TASKS
        for p in self._players.values():
            p.room = SPAWN_ROOM
            p.intent = None
            if p.role is Role.IMPOSTOR:
                p.kill_cooldown = self._kill_cooldown

    def _tick_sabotage(self) -> MatchEnd | None:
        """A crewmate standing in the sabotaged room fixes it — a reflex, as on the server."""
        if self._sabotage is None:
            return None
        system, left = self._sabotage
        room = SABOTAGE_ROOMS[system]
        fixers = [pid for pid in sorted(self.alive()) if self._players[pid].role is Role.CREWMATE and self._players[pid].room == room]
        if fixers:
            self._sabotage = None
            self._emit("SABOTAGE_FIXED", EventKind.SABOTAGE_FIXED, self.alive(), room=room,
                       actor=fixers[0], target=system, exclude_actor=False)
            return None
        if system not in CRITICAL_SABOTAGES:
            return None
        if left <= 1:
            self._sabotage = None
            return self._end("impostors", f"{system} sabotage was not fixed")
        self._sabotage = (system, left - 1)
        return None

    def _check_end(self) -> MatchEnd | None:
        alive = [self._players[pid] for pid in self.alive()]
        impostors = sum(1 for p in alive if p.role is Role.IMPOSTOR)
        crew = len(alive) - impostors
        if impostors == 0:
            return self._end("crew", "all impostors ejected")
        if impostors >= crew:
            return self._end("impostors", "impostors reached parity")
        # A dead crewmate's unfinished tasks no longer block the crew.
        if all(not p.tasks for p in alive if p.role is Role.CREWMATE):
            return self._end("crew", "all tasks completed")
        return None

    def _end(self, winner: str, reason: str) -> MatchEnd:
        self.phase = Phase.ENDED
        self.winner = MatchEnd(winner, reason)
        return self.winner

    def _kill_targets(self, pid: str) -> list[str]:
        return [
            other for other in self._others_in(self._players[pid].room, pid)
            if self._players[other].role is Role.CREWMATE
        ]

    def _others_in(self, room: str, pid: str) -> list[str]:
        return [
            other.player_id for other in self._players.values()
            if other.alive and other.room == room and other.player_id != pid
        ]

    def _emit(
        self,
        objective: str,
        perceived: EventKind,
        witnesses: Sequence[str],
        *,
        room: str | None,
        actor: str | None = None,
        target: str | None = None,
        text: str | None = None,
        evidence_ids: tuple[str, ...] = (),
        exclude_actor: bool = True,
        target_is_player: bool = False,
    ) -> GameEvent:
        """Record what happened, and hand each witness its own view of it.

        Each witness perceives the actor (and a player target) as they appear to it: a disguise, an
        unknown shape in the dark, or not at all — an invisible actor is no event for anyone else.
        """

        event_id = f"e{len(self.events) + 1:04d}"
        views: dict[str, ObservedEvent] = {}
        for pid in witnesses:
            if exclude_actor and pid == actor:
                continue
            seen_actor = self._apparent(actor, pid) if actor in self._players else actor
            seen_target = self._apparent(target, pid) if target_is_player and target in self._players else target
            if seen_actor is _HIDDEN:
                continue
            view = ObservedEvent(
                event_id=event_id, tick=self.tick, kind=perceived, room=room,
                actor=None if seen_actor == SOMEONE else seen_actor,
                target=None if seen_target in (SOMEONE, _HIDDEN) else seen_target,
                text=text, evidence_ids=evidence_ids,
            )
            views[pid] = view
            self._players[pid].pending.append(view)
        event = GameEvent(event_id, self.tick, objective, room, actor, target, text, views)
        self.events.append(event)
        return event

    def _emit_change(
        self, objective: str, perceived: EventKind, player: _Player, witnesses: Sequence[str], *,
        before: str | None, after: str | None, text: str,
    ) -> GameEvent:
        """A visible transformation: witnesses see the shape before and after, never the role."""
        event_id = f"e{len(self.events) + 1:04d}"
        views: dict[str, ObservedEvent] = {}
        for pid in witnesses:
            dark = self._outage("lights") and self._players[pid].role is Role.CREWMATE
            view = ObservedEvent(event_id, self.tick, perceived, player.room,
                                 None if dark else before, None if dark else after, text)
            views[pid] = view
            self._players[pid].pending.append(view)
        event = GameEvent(event_id, self.tick, objective, player.room, player.player_id, after, text, views)
        self.events.append(event)
        return event

    @staticmethod
    def _decision_reason(player: _Player, events: tuple[ObservedEvent, ...]) -> str:
        salient = [e.kind.value for e in events if e.kind in _SALIENT]
        if salient:
            return "observed: " + ", ".join(salient)
        if player.intent is None:
            return "goal reached"
        return "phase changed" if player.phase_seen is not Phase.TASKS else "opportunity"
