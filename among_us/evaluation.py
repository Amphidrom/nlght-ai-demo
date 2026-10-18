"""Measure what the agents did against what actually happened.

Only this module joins the objective event log with the agents' decisions.
The metrics are intentionally separate numbers rather than one score: an ego
that remembers badly is working as designed, an agent that knows what it
never perceived is a defect.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Any

from among_us.ego import load_ego
from among_us.match import MatchRecord
from among_us.memory import consolidate, empty_store, recall
from among_us.protocol import ObservedEvent

_DETAILS = ("room", "actor", "target", "text")


def evaluate(record: MatchRecord) -> dict[str, Any]:
    decisions = [e for e in record.entries if e["type"] == "decision"]
    truth = record.ground_truth
    views: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    if truth is not None:
        for event in truth.events:
            for agent, view in event["views"].items():
                views[agent][event["event_id"]] = view
    return {
        "match": record.summary(),
        "ground_truth_available": truth is not None,
        "memory_leakage": _leakage(decisions, views) if truth is not None else None,
        "recall": _recall(decisions, views) if truth is not None else None,
        "action_validity": _validity(decisions),
        "unsupported_claim_rate": _unsupported(decisions),
        "deception_consistency": "not measured in this slice",
    }


def _leakage(decisions: Sequence[dict[str, Any]], views: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Every memory and cited id must be an event the agent itself perceived."""

    references = violations = 0
    examples: list[dict[str, Any]] = []
    for decision in decisions:
        agent = decision["agent_id"]
        result = decision.get("result", {})
        ids = set(result.get("memory", {}).get("recalled", []))
        ids |= set(result.get("memory", {}).get("encoded", []))
        ids |= set(decision["action"].get("evidence_ids") or [])
        for event_id in ids:
            references += 1
            if event_id not in views.get(agent, {}):
                violations += 1
                if len(examples) < 5:
                    examples.append({"agent_id": agent, "event_id": event_id, "tick": decision["tick"]})
    return {"references": references, "violations": violations,
            "rate": round(violations / references, 4) if references else 0.0, "examples": examples}


def _recall(decisions: Sequence[dict[str, Any]], views: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_agent: dict[str, dict[str, Any]] = {}
    latest: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    ever: dict[str, set[str]] = defaultdict(set)
    egos: dict[str, str] = {}
    for decision in decisions:
        agent = decision["agent_id"]
        egos[agent] = decision["ego"]
        for memory in decision.get("result", {}).get("recalled", []):
            latest[agent][memory["event_id"]] = memory
            ever[agent].add(memory["event_id"])
    for agent in sorted(set(egos)):
        correct = total = 0
        for event_id, memory in latest[agent].items():
            view = views.get(agent, {}).get(event_id)
            if view is None:
                continue
            for name in _DETAILS:
                if name in memory["details"]:
                    total += 1
                    remembered, actual = memory["details"][name], view.get(name)
                    if name == "text":
                        correct += int(str(actual or "").startswith(str(remembered).removesuffix(" …")))
                    else:
                        correct += int(remembered == actual)
        witnessed = len(views.get(agent, {}))
        per_agent[agent] = {
            "ego": egos[agent],
            "witnessed": witnessed,
            "ever_recalled": len(ever[agent]),
            "coverage": round(len(ever[agent]) / witnessed, 3) if witnessed else None,
            "detail_accuracy": round(correct / total, 3) if total else None,
        }
    return per_agent


def _validity(decisions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(decisions)
    accepted = sum(1 for d in decisions if d["accepted"])
    proposals = [d for d in decisions if d.get("result", {}).get("validation", {}).get("status") != "no_proposal"]
    model_accepted = sum(1 for d in proposals if d["result"].get("validation", {}).get("status") == "accepted")
    statuses: dict[str, int] = defaultdict(int)
    for d in decisions:
        statuses[d.get("result", {}).get("validation", {}).get("status", "missing")] += 1
    return {
        "decisions": total,
        "accepted_by_game": accepted,
        "game_acceptance_rate": round(accepted / total, 4) if total else None,
        "model_proposals": len(proposals),
        "model_proposal_acceptance_rate": round(model_accepted / len(proposals), 4) if proposals else None,
        "validation_status": dict(statuses),
    }


def _unsupported(decisions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Share of model claims the validator refused for citing unrecallable memories."""

    claims = unsupported = 0
    for d in decisions:
        validation = d.get("result", {}).get("validation", {})
        if validation.get("status") in (None, "no_proposal"):
            continue
        proposal_cites = validation.get("status") == "unsupported" or d["action"].get("evidence_ids")
        is_statement = d["action"].get("action") == "SAY"
        if proposal_cites or is_statement:
            claims += 1
            unsupported += int(validation.get("status") == "unsupported")
    return {"claims": claims, "unsupported": unsupported,
            "rate": round(unsupported / claims, 4) if claims else None}


def differentiation_probe(
    events: Iterable[ObservedEvent],
    egos: Sequence[str],
    *,
    match_seed: int,
    now: int,
    agent_id: str = "probe",
) -> dict[str, Any]:
    """Feed one identical event stream to several egos and compare what each keeps.

    The agent id is shared on purpose: with the same seed and the same events,
    every difference comes from the profile alone.
    """

    stream = list(events)
    memories: dict[str, dict[str, Any]] = {}
    for ego_id in egos:
        update = consolidate(empty_store(), stream, load_ego(ego_id), match_seed=match_seed, agent_id=agent_id, now=now)
        recalled = recall(update.store, load_ego(ego_id), now)
        memories[ego_id] = {
            "encoded": list(update.encoded),
            "recalled": [m["event_id"] for m in recalled],
            "details": {m["event_id"]: m["details"] for m in recalled},
        }
    distances = {}
    for a, b in combinations(egos, 2):
        left, right = set(memories[a]["recalled"]), set(memories[b]["recalled"])
        union = left | right
        distances[f"{a}~{b}"] = round(1 - len(left & right) / len(union), 3) if union else 0.0
    return {"events": len(stream), "memories": memories, "recall_distance": distances}
