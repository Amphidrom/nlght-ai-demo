"""A seed-deterministic, tick-based Among Us simulation.

It plays the role the C# bridge plays for the real game: it owns the truth,
filters what each player perceives, and accepts only legal actions. Agents
never read the engine; they read the observations it hands out.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from among_us.domain import GameEvent
from among_us.protocol import (
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
from among_us.sim.map import EMERGENCY_BUTTON_ROOM, ROOMS, SPAWN_ROOM, TASKS, next_hop

_SALIENT = frozenset({EventKind.KILL_WITNESSED, EventKind.BODY_FOUND, EventKind.MEETING_CALLED})

MIN_PLAYERS = 4
MAX_PLAYERS = 15


def max_impostors(players: int) -> int:
    """The game's own lobby limit: 1 up to six players, 2 up to eight, then 3."""
    return 1 if players <= 6 else 2 if players <= 8 else 3


@dataclass
class _Player:
    player_id: str
    role: Role
    tasks: list[str]
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
        tasks_per_player: int = 4,
        task_ticks: int = 3,
        kill_cooldown: int = 5,
        meeting_rounds: int = 1,
        max_ticks: int = 240,
    ) -> None:
        if len(set(players)) != len(players) or not MIN_PLAYERS <= len(players) <= MAX_PLAYERS:
            raise ValueError(f"a match needs {MIN_PLAYERS} to {MAX_PLAYERS} distinct players")
        if not 1 <= impostors <= max_impostors(len(players)):
            raise ValueError(f"{len(players)} players allow 1 to {max_impostors(len(players))} impostors")
        self.seed = seed
        self._rng = random.Random(seed)
        order = sorted(players)
        impostor_ids = set(self._rng.sample(order, impostors))
        self._players: dict[str, _Player] = {}
        for pid in order:
            role = Role.IMPOSTOR if pid in impostor_ids else Role.CREWMATE
            # Impostors are dealt a task list too: it is their cover story.
            tasks = self._rng.sample(sorted(TASKS), tasks_per_player)
            self._players[pid] = _Player(pid, role, tasks)
        self._kill_cooldown = kill_cooldown
        self._task_ticks = task_ticks
        self._meeting_rounds = meeting_rounds
        self._max_ticks = max_ticks
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

    def role_of(self, pid: str) -> Role:
        """Ground truth for evaluation. Never part of an observation."""
        return self._players[pid].role

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
        visible = tuple(
            VisiblePlayer(other.player_id, other.room, other.alive)
            for other in self._players.values()
            if other.player_id != pid and other.alive and (in_meeting or other.room == player.room)
        )
        bodies = () if in_meeting else tuple(
            VisibleBody(b.body_id, b.player_id, b.room) for b in self._bodies.values() if b.room == player.room
        )
        return Observation(
            agent_id=pid,
            tick=self.tick,
            phase=self.phase,
            self_state=SelfState(
                player_id=pid,
                room=player.room,
                role=player.role,
                alive=player.alive,
                tasks_remaining=tuple(player.tasks),
                kill_ready=player.role is Role.IMPOSTOR and player.kill_cooldown == 0,
                known_impostors=tuple(
                    other for other, p in self._players.items()
                    if p.role is Role.IMPOSTOR and other != pid
                ) if player.role is Role.IMPOSTOR else (),
            ),
            visible_players=visible,
            visible_bodies=bodies,
            events=events,
            legal_actions=self.legal_actions(pid),
            decision_reason=reason,
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
        actions = [LegalAction(ActionKind.MOVE_TO, room) for room in ROOMS if room != player.room]
        actions += [LegalAction(ActionKind.DO_TASK, t) for t in player.tasks if TASKS[t] == player.room]
        actions += [
            LegalAction(ActionKind.REPORT, b.body_id) for b in self._bodies.values() if b.room == player.room
        ]
        if player.role is Role.IMPOSTOR and player.kill_cooldown == 0:
            actions += [LegalAction(ActionKind.KILL, target) for target in self._kill_targets(pid)]
        if player.room == EMERGENCY_BUTTON_ROOM and not player.button_used:
            actions.append(LegalAction(ActionKind.CALL_MEETING))
        actions.append(LegalAction(ActionKind.WAIT))
        return tuple(actions)

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
        # Kills, reports and the button are instantaneous: they resolve before
        # anyone walks, so a victim cannot step away from a kill already made.
        instant = (ActionKind.KILL, ActionKind.REPORT, ActionKind.CALL_MEETING)
        for first_pass in (True, False):
            for pid in sorted(self.alive()):
                player = self._players[pid]
                is_instant = player.intent is not None and player.intent.action in instant
                if player.alive and self.phase is Phase.TASKS and is_instant == first_pass:
                    self._execute(player)
        for player in self._players.values():
            player.kill_cooldown = max(0, player.kill_cooldown - 1)
        self._record_positions()
        if self.tick >= self._max_ticks and self.phase is Phase.TASKS:
            return self._end("draw", "tick limit reached")
        return self._check_end()

    # -- internals -------------------------------------------------------------

    def _record_positions(self) -> None:
        self.positions[self.tick] = {pid: p.room for pid, p in self._players.items() if p.alive}

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
            if victim and victim.alive and victim.room == player.room and player.kill_cooldown == 0:
                victim.alive = False
                victim.pending = []
                victim.intent = None
                witnesses = [p for p in self._others_in(player.room, player.player_id) if p != victim.player_id]
                self._emit("KILL", EventKind.KILL_WITNESSED, witnesses, room=player.room,
                           actor=player.player_id, target=victim.player_id)
                body_id = f"body_{victim.player_id}"
                self._bodies[body_id] = _Body(body_id, victim.player_id, player.room)
                player.seen_bodies.add(body_id)
                player.kill_cooldown = self._kill_cooldown
                for pid in witnesses:
                    self._players[pid].seen_bodies.add(body_id)
        elif kind is ActionKind.REPORT and intent.target in self._bodies:
            body = self._bodies[intent.target]
            if body.room == player.room:
                self._start_meeting(player, body.player_id, "body reported")
        elif kind is ActionKind.CALL_MEETING and player.room == EMERGENCY_BUTTON_ROOM and not player.button_used:
            player.button_used = True
            self._start_meeting(player, None, "emergency meeting")
        player.intent = None

    def _discover_bodies(self, player: _Player) -> None:
        for body in self._bodies.values():
            if body.room == player.room and body.body_id not in player.seen_bodies:
                player.seen_bodies.add(body.body_id)
                self._emit("BODY_DISCOVERED", EventKind.BODY_FOUND, [player.player_id],
                           room=body.room, actor=None, target=body.player_id)

    def _start_meeting(self, caller: _Player, victim: str | None, reason: str) -> None:
        self._emit("MEETING", EventKind.MEETING_CALLED, self.alive(), room=caller.room,
                   actor=caller.player_id, target=victim, text=reason, exclude_actor=False)
        self._bodies.clear()
        self.phase = Phase.MEETING
        self._round = 0
        self._statements = {}
        self._votes = {}
        for p in self._players.values():
            p.intent = None

    def _resolve_votes(self) -> None:
        tally = Counter(target for target in self._votes.values())
        for voter, target in sorted(self._votes.items()):
            self._emit("VOTE", EventKind.VOTE_CAST, self.alive(), room=None, actor=voter,
                       target=target, exclude_actor=False)
        skip = tally.pop(None, 0)
        ranked = tally.most_common()
        ejected = None
        if ranked and ranked[0][1] > skip and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            ejected = ranked[0][0]
        if ejected:
            self._emit("EJECT", EventKind.EJECTED, self.alive(), room=None, target=ejected, exclude_actor=False)
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
    ) -> GameEvent:
        event_id = f"e{len(self.events) + 1:04d}"
        views: dict[str, ObservedEvent] = {}
        for pid in witnesses:
            if exclude_actor and pid == actor:
                continue
            view = ObservedEvent(
                event_id=event_id, tick=self.tick, kind=perceived, room=room,
                actor=actor, target=target, text=text, evidence_ids=evidence_ids,
            )
            views[pid] = view
            self._players[pid].pending.append(view)
        event = GameEvent(event_id, self.tick, objective, room, actor, target, text, views)
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
