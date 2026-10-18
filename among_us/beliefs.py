"""Subjective suspicion, derived from what an agent can recall right now.

Beliefs are recomputed from recalled memories on every turn instead of being
accumulated, so forgetting a sighting also weakens the suspicion it carried.
The update is deterministic and every belief lists the memories behind it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from among_us.ego import EgoProfile
from among_us.protocol import EventKind

_PRIOR = 0.2
_NEAR_BODY_WINDOW = 6


def update_beliefs(
    recalled: Iterable[dict[str, Any]],
    ego: EgoProfile,
    *,
    self_id: str,
    players: Iterable[str],
    partners: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    memories = list(recalled)
    partners = set(partners)
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for pid in sorted(set(players) - {self_id}):
        scores[pid] = 0.0 if pid in partners else _PRIOR
        evidence[pid] = []

    def bump(pid: str | None, delta: float, memory: dict[str, Any]) -> None:
        if pid not in scores or pid in partners:
            return
        weighted = delta * float(memory["certainty"]) * (ego.judgement.suspicion_bias if delta > 0 else 1.0)
        scores[pid] += weighted
        evidence[pid].append(memory["event_id"])

    bodies = [m for m in memories if m["kind"] in (EventKind.BODY_FOUND.value, EventKind.MEETING_CALLED.value)]
    for memory in memories:
        kind = memory["kind"]
        details = memory["details"]
        actor = details.get("actor")
        if kind == EventKind.KILL_WITNESSED.value:
            bump(actor, 1.0, memory)
        elif kind == EventKind.VENT_SEEN.value:
            # Only impostors vent: seeing it is nearly as damning as seeing a kill.
            bump(actor, 0.9, memory)
        elif kind == EventKind.TASK_SEEN.value:
            bump(actor, -0.08, memory)
        elif kind in (EventKind.PLAYER_LEFT.value, EventKind.PLAYER_ENTERED.value):
            for body in bodies:
                same_room = details.get("room") and details.get("room") == body["details"].get("room")
                if same_room and 0 <= int(body["tick"]) - int(memory["tick"]) <= _NEAR_BODY_WINDOW:
                    bump(actor, 0.25, memory)
        elif kind == EventKind.CHAT.value and actor != self_id:
            text = str(details.get("text", "")).lower()
            for pid in scores:
                if pid.lower() in text and any(w in text for w in ("sus", "suspect", "saw", "kill", "vote")):
                    bump(pid, 0.05, memory)
        elif kind == EventKind.VOTE_CAST.value and details.get("target") == self_id:
            # Being voted for is not evidence about the voter's role, but it is
            # remembered: it makes the voter a little suspicious to this agent.
            bump(actor, 0.05, memory)
    return {
        pid: {"suspicion": round(min(1.0, max(0.0, score)), 3), "evidence_ids": sorted(set(evidence[pid]))}
        for pid, score in scores.items()
    }


def most_suspected(beliefs: dict[str, dict[str, Any]], *, alive: Iterable[str], threshold: float = 0.6) -> str | None:
    candidates = [(b["suspicion"], pid) for pid, b in beliefs.items() if pid in set(alive)]
    if not candidates:
        return None
    score, pid = max(candidates)
    return pid if score >= threshold else None
