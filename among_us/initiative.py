"""Acting on one's own: what an agent does between decisions because of what it believes.

The game asks for a decision at its decision points. Between them every observation still updates
memory and beliefs, and a belief can grow strong enough to act on before the game asks. This module
decides that — deterministically, per ego, without the model:

- A crewmate whose suspicion of a living player reaches its ego's ``call_meeting_at`` calls an
  emergency meeting and cites the memories behind the suspicion — but only for hard evidence it saw
  itself since the last meeting (a kill, a vent, a change of shape), never for what was said or
  concluded at the last one. The same memories never trigger a second call.

Only an action the observation lists as legal is proposed; the server checks it once more against
the game as it is when the action arrives.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from among_us.assess import fresh_evidence
from among_us.ego import EgoProfile
from among_us.protocol import ActionKind, EventKind, LegalAction, Observation, Phase, Role, SemanticAction

#: Hive Mind slot: suspect → the evidence ids the agent already acted on, for this match.
INITIATIVE_SLOT = "among_us:initiative"

_DEATHS = (EventKind.BODY_FOUND.value, EventKind.KILL_WITNESSED.value, EventKind.EJECTED.value)


def initiative_action(
    observation: Observation,
    beliefs: dict[str, dict[str, Any]],
    recalled: Iterable[dict[str, Any]],
    ego: EgoProfile,
    acted: dict[str, list[str]],
) -> tuple[SemanticAction | None, dict[str, list[str]], dict[str, Any] | None]:
    """The action the agent takes on its own now, if any, the updated ``acted`` record, and why."""

    me = observation.self_state
    if (
        observation.phase is not Phase.TASKS
        or me.role is not Role.CREWMATE
        or not me.alive
        or LegalAction(ActionKind.CALL_MEETING) not in observation.legal_actions
    ):
        return None, acted, None
    memories = list(recalled)
    recalled_ids = {m["event_id"] for m in memories}
    dead = {m["details"].get("target") for m in memories if m["kind"] in _DEATHS}
    candidates = sorted(
        ((belief["suspicion"], pid) for pid, belief in beliefs.items()
         if pid not in dead and belief["suspicion"] >= ego.judgement.call_meeting_at),
        reverse=True,
    )
    for suspicion, suspect in candidates:
        evidence = sorted(set(beliefs[suspect]["evidence_ids"]) & recalled_ids)
        if not set(fresh_evidence(observation, memories, evidence)) - set(acted.get(suspect, [])):
            continue
        action = SemanticAction(observation.agent_id, ActionKind.CALL_MEETING, goal="ACCUSE", evidence_ids=tuple(evidence))
        updated = {**acted, suspect: sorted(set(acted.get(suspect, [])) | set(evidence))}
        return action, updated, {"suspect": suspect, "suspicion": suspicion, "threshold": ego.judgement.call_meeting_at}
    return None, acted, None
