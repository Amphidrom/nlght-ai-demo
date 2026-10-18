"""Ego-dependent episodic memory: attention, encoding, retention, recall.

The objective events are never changed. What changes is what an agent keeps of
its *own* view of them: whether it noticed an event at all, which details it
encoded and how faithfully, and how long the trace stays recallable. Every
choice draws from a generator seeded by (match, agent, event, purpose), so the
same match replays to the same memories, and an agent's errors come from its
profile — never from information it did not perceive.

Memory content is JSON so it survives in a Hive Mind coordinator slot.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from among_us.ego import EgoProfile
from among_us.protocol import EventKind, ObservedEvent
from among_us.sim.map import ADJACENCY

_DETAILS = ("room", "actor", "target", "text")
_SALIENT = frozenset({EventKind.KILL_WITNESSED, EventKind.BODY_FOUND})


def rng_for(match_seed: int, agent_id: str, event_id: str, purpose: str) -> random.Random:
    digest = hashlib.sha256(f"{match_seed}|{agent_id}|{event_id}|{purpose}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def empty_store() -> dict[str, Any]:
    return {"memories": {}, "processed": []}


def encode(event: ObservedEvent, ego: EgoProfile, *, match_seed: int, agent_id: str) -> dict[str, Any] | None:
    """The trace this ego keeps of one perceived event, or ``None`` if unnoticed."""

    if rng_for(match_seed, agent_id, event.event_id, "attention").random() >= ego.attention_for(event.kind):
        return None
    fidelity = ego.encoding.fidelity
    if event.kind in _SALIENT:
        fidelity = min(1.0, fidelity + 0.3)
    rng = rng_for(match_seed, agent_id, event.event_id, "encoding")
    available = [name for name in _DETAILS if getattr(event, name)]
    details: dict[str, Any] = {}
    for name in available:
        if rng.random() >= fidelity:
            continue
        value = getattr(event, name)
        if name == "room" and value in ADJACENCY and rng.random() < ego.encoding.detail_noise:
            # A plausible mix-up with a neighbouring room — the agent cannot tell.
            value = rng.choice(sorted(ADJACENCY[value]))
        if name == "text":
            words = str(value).split()
            keep = max(1, round(len(words) * fidelity))
            value = " ".join(words[:keep]) + (" …" if keep < len(words) else "")
        details[name] = value
    if event.evidence_ids and "text" in details:
        details["cited"] = list(event.evidence_ids)
    quality = (len(details) / len(available)) if available else 1.0
    return {
        "event_id": event.event_id,
        "tick": event.tick,
        "kind": event.kind.value,
        "details": details,
        "encoding_quality": round(quality, 3),
    }


def strength(memory: dict[str, Any], ego: EgoProfile, now: int) -> float:
    age = max(0, now - int(memory["tick"]))
    salience = 2.0 if memory["kind"] in {k.value for k in _SALIENT} else 1.0
    decay = 0.5 ** (age / (ego.retention.half_life_ticks * salience))
    return float(memory["encoding_quality"]) * decay


@dataclass(frozen=True)
class MemoryUpdate:
    store: dict[str, Any]
    encoded: tuple[str, ...]
    ignored: tuple[str, ...]
    forgotten: tuple[str, ...]


def consolidate(
    store: dict[str, Any],
    events: Iterable[ObservedEvent],
    ego: EgoProfile,
    *,
    match_seed: int,
    agent_id: str,
    now: int,
) -> MemoryUpdate:
    """Encode new perceptions and drop traces that have faded for good.

    A forgotten trace is deleted, not hidden: it cannot resurface later.
    """

    memories = dict(store.get("memories", {}))
    processed = list(store.get("processed", []))
    seen = set(processed)
    encoded: list[str] = []
    ignored: list[str] = []
    for event in events:
        if event.event_id in seen:
            continue
        seen.add(event.event_id)
        processed.append(event.event_id)
        trace = encode(event, ego, match_seed=match_seed, agent_id=agent_id)
        if trace is None:
            ignored.append(event.event_id)
        else:
            memories[event.event_id] = trace
            encoded.append(event.event_id)
    forget_below = ego.retention.recall_threshold * 0.5
    forgotten = sorted(eid for eid, m in memories.items() if strength(m, ego, now) < forget_below)
    for eid in forgotten:
        del memories[eid]
    return MemoryUpdate({"memories": memories, "processed": processed}, tuple(encoded), tuple(ignored), tuple(forgotten))


def recall(store: dict[str, Any], ego: EgoProfile, now: int) -> list[dict[str, Any]]:
    """Traces strong enough to come to mind now, newest first, with certainty."""

    recalled = []
    for memory in store.get("memories", {}).values():
        value = strength(memory, ego, now)
        if value >= ego.retention.recall_threshold:
            recalled.append({**memory, "certainty": round(value, 3)})
    return sorted(recalled, key=lambda m: (-int(m["tick"]), m["event_id"]))
