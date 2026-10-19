"""The deterministic fallback every agent can always take.

It runs when no model is involved (a single legal action, an offline match) and
when the validator refuses a model proposal. It reads only the agent's own
observation, recalled memories and beliefs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from among_us.agenda import questions_for
from among_us.beliefs import most_suspected
from among_us.protocol import ActionKind, EventKind, Observation, Phase, Role, SemanticAction
from among_us.sim.map import ROOMS, SABOTAGE_ROOMS, TASKS

_HUNTING_GROUNDS = ("electrical", "admin", "medbay", "storage", "navigation", "weapons")
_ACCUSING = ("sus", "suspect", "saw", "kill", "vote", "impostor", "lying", "liar")
_QUESTIONS = (
    "Where was everyone when it happened?",
    "Who found the body, and where exactly?",
    "Can anyone vouch for where they were?",
)
_PRIOR = 0.2


def fallback_action(observation: Observation, beliefs: dict[str, dict[str, Any]], recalled_ids: set[str],
                    agenda: Sequence[dict[str, Any]] = ()) -> SemanticAction:
    me = observation.self_state
    legal = {(a.action, a.target) for a in observation.legal_actions}
    agent = observation.agent_id

    def legal_targets(kind: ActionKind) -> list[str]:
        return sorted(t for k, t in legal if k is kind and t is not None)

    if observation.phase is Phase.MEETING or (observation.phase is Phase.VOTING and (ActionKind.SAY, None) in legal):
        # While the table can still talk, talk (or wait); the vote comes with the last call.
        return _meeting_turn(observation, beliefs, recalled_ids, agenda)

    if observation.phase is Phase.VOTING:
        candidates = [t for t in legal_targets(ActionKind.VOTE) if t not in me.known_impostors]
        suspect = most_suspected({p: b for p, b in beliefs.items() if p in candidates}, alive=candidates)
        if suspect:
            return SemanticAction(agent, ActionKind.VOTE, target=suspect, goal="EJECT",
                                  evidence_ids=tuple(sorted(set(beliefs[suspect]["evidence_ids"]) & recalled_ids)))
        return SemanticAction(agent, ActionKind.SKIP_VOTE, goal="WAIT_FOR_EVIDENCE")

    just_killed = any(e.kind is EventKind.KILL_COMMITTED for e in observation.events)
    if me.role is Role.IMPOSTOR:
        if just_killed:
            # Leave the scene: through a vent if there is one, else on foot.
            vents = legal_targets(ActionKind.VENT)
            if vents:
                return SemanticAction(agent, ActionKind.VENT, target=vents[observation.tick % len(vents)], goal="ESCAPE")
        kills = legal_targets(ActionKind.KILL)
        if len(kills) == 1 and len(observation.visible_players) == 1:
            return SemanticAction(agent, ActionKind.KILL, target=kills[0], goal="ELIMINATE")
        if me.vanished and (ActionKind.APPEAR, None) in legal and observation.visible_players:
            return SemanticAction(agent, ActionKind.APPEAR, goal="STRIKE")
        if (ActionKind.VANISH, None) in legal and observation.visible_players and observation.tick % 4 == 0:
            return SemanticAction(agent, ActionKind.VANISH, goal="SLIP_AWAY")
        shapes = legal_targets(ActionKind.SHAPESHIFT)
        if shapes and not observation.visible_players:
            # Take on the look of a crewmate, so a witness blames them.
            crew = [p for p in shapes if p not in me.known_impostors] or shapes
            return SemanticAction(agent, ActionKind.SHAPESHIFT, target=crew[observation.tick % len(crew)], goal="FRAME")
        sabotages = legal_targets(ActionKind.SABOTAGE)
        if sabotages and not observation.visible_players and observation.tick % 3 == 0:
            return SemanticAction(agent, ActionKind.SABOTAGE, target=sabotages[observation.tick % len(sabotages)], goal="SPLIT_CREW")
        doors = legal_targets(ActionKind.CLOSE_DOORS)
        if doors and observation.tick % 5 == 0:
            return SemanticAction(agent, ActionKind.CLOSE_DOORS, target=doors[observation.tick % len(doors)], goal="TRAP")
        if me.kill_ready:
            # Hunt: roam the busy rooms instead of faking tasks.
            hunting = [room for room in _HUNTING_GROUNDS if (ActionKind.MOVE_TO, room) in legal]
            if hunting:
                return SemanticAction(agent, ActionKind.MOVE_TO, target=hunting[observation.tick % len(hunting)], goal="HUNT")
    else:
        reports = legal_targets(ActionKind.REPORT)
        if reports:
            return SemanticAction(agent, ActionKind.REPORT, target=reports[0], goal="REPORT_BODY")
        for system in (observation.alarm, *observation.outages):
            if system and (ActionKind.MOVE_TO, SABOTAGE_ROOMS.get(system)) in legal:
                return SemanticAction(agent, ActionKind.MOVE_TO, target=SABOTAGE_ROOMS[system], goal="FIX_SABOTAGE")
    tasks = legal_targets(ActionKind.DO_TASK)
    if tasks:
        return SemanticAction(agent, ActionKind.DO_TASK, target=tasks[0], goal="TASK")
    # Task rooms are known for the simulated ship only; the real game names its tasks differently.
    for task in me.tasks_remaining:
        if task in TASKS and (ActionKind.MOVE_TO, TASKS[task]) in legal:
            return SemanticAction(agent, ActionKind.MOVE_TO, target=TASKS[task], goal="TASK")
    # Nothing left to do: wander deterministically so the agent stays observable.
    rooms = [room for room in ROOMS if (ActionKind.MOVE_TO, room) in legal]
    if rooms:
        return SemanticAction(agent, ActionKind.MOVE_TO, target=rooms[observation.tick % len(rooms)], goal="PATROL")
    return SemanticAction(agent, ActionKind.WAIT)


def _meeting_turn(observation: Observation, beliefs: dict[str, dict[str, Any]], recalled_ids: set[str],
                  agenda: Sequence[dict[str, Any]] = ()) -> SemanticAction:
    """A meeting line without the model: answer what it was asked, work through its agenda, answer an accusation,
    accuse with evidence, take a side, or ask.

    Never the same line twice in one meeting; with nothing new to say, wait.
    """

    me, agent = observation.self_state, observation.agent_id
    chat = observation.meeting_chat
    said = {m.text for m in chat if m.speaker == agent}
    alive = [p.player_id for p in observation.visible_players]
    partners = set(me.known_impostors)

    def accuses(text: str, player: str) -> bool:
        lower = text.lower()
        return player in lower and any(word in lower for word in _ACCUSING)

    lines: list[tuple[str, str, tuple[str, ...]]] = []
    for question in questions_for(observation):
        lines.append((f"{question.asker}, I was in {me.room.replace('_', ' ')}.", "ANSWER", ()))
    for item in agenda:
        if not item["done"]:
            lines.append((item["text"], "TELL" if item["kind"] == "tell" else "QUESTION",
                          tuple(e for e in item["evidence"] if e in recalled_ids)))
    accusers = [m.speaker for m in chat if m.speaker != agent and accuses(m.text, agent)]
    if accusers:
        lines.append((f"{accusers[-1]}, it wasn't me. I was in {me.room.replace('_', ' ')}.", "DEFEND", ()))
    suspect = most_suspected(beliefs, alive=alive)
    if me.role is Role.CREWMATE and suspect:
        cited = tuple(sorted(set(beliefs[suspect]["evidence_ids"]) & recalled_ids))
        if cited:
            lines.append((f"I suspect {suspect}. I saw something.", "ACCUSE", cited))
    for message in reversed(chat):
        named = [p for p in alive if p != message.speaker and p not in partners and accuses(message.text, p)]
        if message.speaker == agent or not named:
            continue
        target = named[0]
        if beliefs.get(target, {}).get("suspicion", _PRIOR) > _PRIOR:
            lines.append((f"I agree with {message.speaker} about {target}.", "AGREE", ()))
        else:
            lines.append((f"{message.speaker}, what makes you sure about {target}?", "QUESTION", ()))
        break
    lines.append((_QUESTIONS[(len(chat) + len(agent)) % len(_QUESTIONS)], "QUESTION", ()))
    for text, goal, evidence in lines:
        if text not in said:
            return SemanticAction(agent, ActionKind.SAY, text=text, goal=goal, evidence_ids=evidence)
    return SemanticAction(agent, ActionKind.WAIT)
