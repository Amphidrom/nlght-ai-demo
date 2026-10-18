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
    recall_report = _recall(decisions, views) if truth is not None else None
    return {
        "match": record.summary(),
        "ground_truth_available": truth is not None,
        "memory_leakage": _leakage(decisions, views) if truth is not None else None,
        "recall": recall_report,
        "by_memory_mode": _by_mode(decisions, recall_report, truth.roles if truth else None),
        "action_validity": _validity(decisions),
        "unsupported_claim_rate": _unsupported(decisions),
        "claims": _claims(decisions, truth) if truth is not None and truth.positions else None,
        "impostor_play": _impostor_play(truth, record.winner) if truth is not None else None,
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


def _by_mode(
    decisions: Sequence[dict[str, Any]],
    recall_report: dict[str, dict[str, Any]] | None,
    roles: dict[str, str] | None,
) -> dict[str, Any]:
    """Ego memory against the transcript baseline, on the same match.

    Crewmate votes are scored against the true roles; impostor votes are
    strategic and left out. Context size is what each model turn had to read.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        groups[decision.get("memory_mode", "ego")].append(decision)
    report: dict[str, Any] = {}
    for mode, items in sorted(groups.items()):
        agents = sorted({d["agent_id"] for d in items})
        sizes = [d["result"]["context"]["chars"] for d in items if d.get("result", {}).get("context")]
        crew_votes = [
            d for d in items
            if d["phase"] == "VOTING" and d["accepted"] and roles and roles.get(d["agent_id"]) == "CREWMATE"
        ]
        on_target = [d for d in crew_votes if d["action"]["action"] == "VOTE"]
        correct = sum(1 for d in on_target if roles and roles.get(d["action"]["target"]) == "IMPOSTOR")
        recall_rows = [recall_report[a] for a in agents if recall_report and a in recall_report]
        coverage = [r["coverage"] for r in recall_rows if r["coverage"] is not None]
        accuracy = [r["detail_accuracy"] for r in recall_rows if r["detail_accuracy"] is not None]
        report[mode] = {
            "agents": agents,
            "decisions": len(items),
            "model_turns": len(sizes),
            "context_chars_mean": round(sum(sizes) / len(sizes)) if sizes else None,
            "context_chars_max": max(sizes) if sizes else None,
            "crew_votes": len(crew_votes),
            "crew_skip_rate": round(1 - len(on_target) / len(crew_votes), 3) if crew_votes else None,
            "crew_vote_accuracy": round(correct / len(on_target), 3) if on_target else None,
            "recall_coverage_mean": round(sum(coverage) / len(coverage), 3) if coverage else None,
            "detail_accuracy_mean": round(sum(accuracy) / len(accuracy), 3) if accuracy else None,
            "unsupported_claims": _unsupported(items),
        }
    return report


def _impostor_play(truth: Any, winner: dict[str, Any] | None) -> dict[str, Any]:
    """How the impostors played: kills and who saw them, vents and who saw them, sabotage, cover."""

    events = truth.events
    kills = [e for e in events if e["kind"] == "KILL"]
    seen_kills = [e for e in kills if set(e["views"]) - {e["actor"]}]
    vents = [e for e in events if e["kind"] == "VENT" and e["text"] == "entered"]
    vent_sightings = sum(len(e["views"]) for e in events if e["kind"] == "VENT")
    sabotages = [e for e in events if e["kind"] == "SABOTAGE"]
    fixed = [e for e in events if e["kind"] == "SABOTAGE_FIXED"]
    killer_of = {e["target"]: e["actor"] for e in kills}
    self_reports = [
        e for e in events
        if e["kind"] == "MEETING" and e["target"] is not None and killer_of.get(e["target"]) == e["actor"]
    ]
    reason = (winner or {}).get("reason", "")
    return {
        "kills": len(kills),
        "kills_witnessed": len(seen_kills),
        "vent_uses": len(vents),
        "vent_sightings": vent_sightings,
        "sabotages": len(sabotages),
        "sabotages_fixed": len(fixed),
        "won_by_sabotage": "sabotage" in reason,
        "self_reports": len(self_reports),
    }


_CLAIM_TOLERANCE = 1  # ticks: "I was in admin at 12" holds if true at 11, 12 or 13


def _claims(decisions: Sequence[dict[str, Any]], truth: Any) -> dict[str, Any]:
    """Score structured location claims against where everyone really was.

    A false claim by an impostor is a lie; by a crewmate it is a mistake —
    typically a memory error, which an ego is allowed to make. A lie is
    detectable if another living player was, at that tick, in the claimed room
    (and did not see the subject) or in the room the subject really was in.
    A speaker contradicts itself by placing the same player at the same tick
    in two different rooms over the match.
    """

    positions: dict[int, dict[str, str]] = truth.positions
    roles: dict[str, str] = truth.roles
    rows: list[dict[str, Any]] = []
    said: dict[str, dict[tuple[str, int], set[str]]] = defaultdict(lambda: defaultdict(set))
    for decision in decisions:
        speaker = decision["agent_id"]
        for claim in decision.get("result", {}).get("claims", []):
            player, room, tick = claim["player"], claim["room"], int(claim["tick"])
            window = range(tick - _CLAIM_TOLERANCE, tick + _CLAIM_TOLERANCE + 1)
            actual = {positions[t].get(player) for t in window if t in positions} - {None}
            true = room in actual
            detectable = False
            if not true and tick in positions:
                real = positions[tick].get(player)
                detectable = any(
                    where in (room, real)
                    for witness, where in positions[tick].items()
                    if witness not in (speaker, player)
                )
            said[speaker][(player, tick)].add(room)
            rows.append({
                "speaker": speaker,
                "role": roles.get(speaker),
                "memory_mode": decision.get("memory_mode", "ego"),
                "about_self": player == speaker,
                "true": true,
                "detectable": detectable,
            })

    def summarise(items: list[dict[str, Any]]) -> dict[str, Any]:
        false = [r for r in items if not r["true"]]
        return {
            "claims": len(items),
            "true": len(items) - len(false),
            "false": len(false),
            "false_about_self": sum(1 for r in false if r["about_self"]),
            "detectable_false": sum(1 for r in false if r["detectable"]),
        }

    impostor_rows = [r for r in rows if r["role"] == "IMPOSTOR"]
    crew_rows = [r for r in rows if r["role"] == "CREWMATE"]
    contradictions = {
        speaker: sum(1 for rooms in claims.values() if len(rooms) > 1)
        for speaker, claims in said.items()
    }
    impostor_keys = sum(len(said[s]) for s in said if roles.get(s) == "IMPOSTOR")
    impostor_contradictions = sum(n for s, n in contradictions.items() if roles.get(s) == "IMPOSTOR")
    lies = summarise(impostor_rows)
    return {
        "impostors": {
            **lies,
            "lies": lies["false"],
            "detectable_lie_rate": round(lies["detectable_false"] / lies["false"], 3) if lies["false"] else None,
            "self_contradictions": impostor_contradictions,
            "deception_consistency": round(1 - impostor_contradictions / impostor_keys, 3) if impostor_keys else None,
        },
        "crew": {
            **summarise(crew_rows),
            "by_memory_mode": {
                mode: summarise([r for r in crew_rows if r["memory_mode"] == mode])
                for mode in sorted({r["memory_mode"] for r in crew_rows})
            },
        },
        "self_contradictions_by_speaker": {s: n for s, n in sorted(contradictions.items()) if n},
    }


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
