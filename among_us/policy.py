"""The deterministic fallback every agent can always take.

It runs when no model is involved (a single legal action, an offline match) and
when the validator refuses a model proposal. It reads only the agent's own
observation, recalled memories and beliefs.
"""

from __future__ import annotations

from typing import Any

from among_us.beliefs import most_suspected
from among_us.protocol import ActionKind, EventKind, Observation, Phase, Role, SemanticAction
from among_us.sim.map import ROOMS, SABOTAGE_ROOMS, TASKS

_HUNTING_GROUNDS = ("electrical", "admin", "medbay", "storage", "navigation", "weapons")


def fallback_action(observation: Observation, beliefs: dict[str, dict[str, Any]], recalled_ids: set[str]) -> SemanticAction:
    me = observation.self_state
    legal = {(a.action, a.target) for a in observation.legal_actions}
    agent = observation.agent_id

    def legal_targets(kind: ActionKind) -> list[str]:
        return sorted(t for k, t in legal if k is kind and t is not None)

    if observation.phase is Phase.MEETING:
        suspect = most_suspected(beliefs, alive=[p.player_id for p in observation.visible_players])
        if me.role is Role.CREWMATE and suspect:
            cited = sorted(set(beliefs[suspect]["evidence_ids"]) & recalled_ids)
            if cited:
                return SemanticAction(agent, ActionKind.SAY, text=f"I suspect {suspect}.", goal="ACCUSE", evidence_ids=tuple(cited))
        return SemanticAction(agent, ActionKind.SAY, text="I have nothing solid to report.", goal="INFORM")

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
        sabotages = legal_targets(ActionKind.SABOTAGE)
        if sabotages and not observation.visible_players and observation.tick % 3 == 0:
            return SemanticAction(agent, ActionKind.SABOTAGE, target=sabotages[observation.tick % len(sabotages)], goal="SPLIT_CREW")
        if me.kill_ready:
            # Hunt: roam the busy rooms instead of faking tasks.
            hunting = [room for room in _HUNTING_GROUNDS if (ActionKind.MOVE_TO, room) in legal]
            if hunting:
                return SemanticAction(agent, ActionKind.MOVE_TO, target=hunting[observation.tick % len(hunting)], goal="HUNT")
    else:
        reports = legal_targets(ActionKind.REPORT)
        if reports:
            return SemanticAction(agent, ActionKind.REPORT, target=reports[0], goal="REPORT_BODY")
        if observation.alarm and (ActionKind.MOVE_TO, SABOTAGE_ROOMS.get(observation.alarm)) in legal:
            return SemanticAction(agent, ActionKind.MOVE_TO, target=SABOTAGE_ROOMS[observation.alarm], goal="FIX_SABOTAGE")
    tasks = legal_targets(ActionKind.DO_TASK)
    if tasks:
        return SemanticAction(agent, ActionKind.DO_TASK, target=tasks[0], goal="TASK")
    for task in me.tasks_remaining:
        if (ActionKind.MOVE_TO, TASKS[task]) in legal:
            return SemanticAction(agent, ActionKind.MOVE_TO, target=TASKS[task], goal="TASK")
    # Nothing left to do: wander deterministically so the agent stays observable.
    rooms = [room for room in ROOMS if (ActionKind.MOVE_TO, room) in legal]
    if rooms:
        return SemanticAction(agent, ActionKind.MOVE_TO, target=rooms[observation.tick % len(rooms)], goal="PATROL")
    return SemanticAction(agent, ActionKind.WAIT)
