"""What an agent wants to bring to the next meeting: things to tell and things to ask.

Kept during the round, as it happens — every turn adds what the agent's memories give, worded as the
line it would say — so at the table the agent can be direct: answer what it was asked, say what it
saw, ask what it needs to know, one point at a time. An item rests on memories; when the agent
forgets them, the item goes too.

At the table the chat is read for what concerns the agent: questions put to it, and the answers
to its own questions, each set against what it saw. Nothing here needs the model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from among_us.assess import _SAME_MOMENT, moment, room_claims
from among_us.protocol import Observation, Role
from among_us.sim.map import ROOMS

#: Hive Mind slot: the agent's open and settled items, per match.
AGENDA_SLOT = "among_us:agenda"

#: Ticks before a body was found in which someone seen in that room is worth asking about.
_NEAR_BODY = 6
_SIGHTINGS = ("PLAYER_ENTERED", "PLAYER_LEFT", "TASK_SEEN")


@dataclass(frozen=True)
class Question:
    asker: str
    text: str


@dataclass(frozen=True)
class Reply:
    player: str
    text: str
    #: "fits", "contradicts" or "" (nothing the agent saw to set it against).
    verdict: str
    seen: str | None = None


def update(items: list[dict[str, Any]], observation: Observation, recalled: Sequence[dict[str, Any]],
           beliefs: dict[str, dict[str, Any]], dead: Sequence[str] = ()) -> list[dict[str, Any]]:
    """The agenda after this turn: new items from what the agent recalls, items on forgotten memories or about the
    dead dropped, and — in a meeting — items it has spoken about settled."""

    recalled_ids = {m["event_id"] for m in recalled}
    gone = set(dead)
    known = {item["id"]: item for item in items
             if (set(item["evidence"]) & recalled_ids or not item["evidence"]) and item["about"] not in gone}
    for item in _items(observation, recalled, beliefs):
        if item["about"] not in gone:
            known.setdefault(item["id"], item)
    said = [m.text.lower() for m in observation.meeting_chat if m.speaker == observation.agent_id]
    for item in known.values():
        if not item["done"] and any(_covers(line, item["text"]) for line in said):
            item["done"] = True
    return sorted(known.values(), key=lambda i: (i["done"], i["rank"], i["id"]))


def questions_for(observation: Observation) -> list[Question]:
    """Questions others put to this agent since it last spoke: answer them first."""

    me = observation.agent_id
    chat = list(observation.meeting_chat)
    last_own = max((i for i, m in enumerate(chat) if m.speaker == me), default=-1)
    return [Question(m.speaker, m.text) for m in chat[last_own + 1:]
            if m.speaker != me and "?" in m.text and _mentions(m.text.lower(), me)]


def replies(items: Sequence[dict[str, Any]], observation: Observation, recalled: Sequence[dict[str, Any]]) -> list[Reply]:
    """What those the agent asked said afterwards, set against what it saw them do."""

    me = observation.agent_id
    chat = list(observation.meeting_chat)
    asked = {item["about"] for item in items if item["kind"] == "ask" and item["done"]}
    found = []
    for player in sorted(asked):
        asked_at = next((i for i, m in enumerate(chat) if m.speaker == me and _mentions(m.text.lower(), player)), None)
        if asked_at is None:
            continue
        answer = next((m for m in chat[asked_at + 1:] if m.speaker == player), None)
        if answer is None:
            continue
        rooms = set(ROOMS) | {m["details"]["room"] for m in recalled if m["details"].get("room")}
        claimed = [room for who, room in room_claims(answer.text, player, observation.roster, rooms) if who == player]
        # Only what the agent saw at the moment in question speaks for or against the answer.
        at = moment(observation, recalled)
        seen = next((m["details"]["room"] for m in sorted(recalled, key=lambda m: -int(m["tick"]))
                     if m["kind"] in _SIGHTINGS and m["details"].get("actor") == player and m["details"].get("room")
                     and at - _SAME_MOMENT <= int(m["tick"]) <= at), None)
        verdict = "" if not claimed or seen is None else "fits" if seen in claimed else "contradicts"
        found.append(Reply(player, answer.text, verdict, seen))
    return found


def _covers(line: str, point: str) -> bool:
    """Whether a line the agent said makes its point: at least half the point's words, however phrased around them."""

    words = {w for w in re.findall(r"[a-z]+", point.lower()) if len(w) > 3}
    return bool(words) and len(words & set(re.findall(r"[a-z]+", line.lower()))) >= len(words) / 2


def _mentions(text: str, player: str) -> bool:
    return re.search(rf"\b{re.escape(player)}\b", text, re.IGNORECASE) is not None


def _room(room: str | None) -> str:
    return (room or "somewhere").replace("_", " ")


def _items(observation: Observation, recalled: Sequence[dict[str, Any]], beliefs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    me = observation.self_state
    partners = set(me.known_impostors)
    impostor = me.role is Role.IMPOSTOR
    items: list[dict[str, Any]] = []

    def add(kind: str, about: str, text: str, evidence: Sequence[str], rank: int) -> None:
        if about == observation.agent_id or about in partners:
            return
        items.append({"id": f"{kind}:{about}:{':'.join(sorted(evidence)) or text}", "kind": kind, "about": about,
                      "text": text, "evidence": sorted(evidence), "rank": rank, "done": False})

    bodies = [m for m in recalled if m["kind"] == "BODY_FOUND"]
    for memory in recalled:
        d, eid = memory["details"], memory["event_id"]
        actor, target, room = d.get("actor"), d.get("target"), d.get("room")
        if not actor or actor == "someone":
            continue
        if memory["kind"] == "KILL_WITNESSED" and not impostor:
            add("tell", actor, f"I saw {actor} kill {target or 'someone'} in {_room(room)}.", [eid], 0)
        elif memory["kind"] == "VENT_SEEN" and not impostor:
            add("tell", actor, f"I saw {actor} {d.get('text') or 'use'} a vent in {_room(room)}.", [eid], 1)
        elif memory["kind"] == "SHAPESHIFT_SEEN" and not impostor:
            add("tell", actor, f"I saw {actor} change shape into {target} in {_room(room)}.", [eid], 1)
        elif memory["kind"] in ("VANISH_SEEN", "APPEAR_SEEN") and not impostor:
            add("tell", actor, f"I saw {actor} {'vanish' if memory['kind'] == 'VANISH_SEEN' else 'appear out of nowhere'} in {_room(room)}.", [eid], 1)
        elif memory["kind"] in _SIGHTINGS and room:
            for body in bodies:
                gap = int(body["tick"]) - int(memory["tick"])
                if body["details"].get("room") == room and 0 <= gap <= _NEAR_BODY:
                    victim = body["details"].get("target") or "someone"
                    # True for the crew; for an impostor a convenient truth that points elsewhere.
                    add("tell", actor, f"I saw {actor} in {_room(room)} just before {victim}'s body was found there.",
                        [eid, body["event_id"]], 2)
                    add("ask", actor, f"{actor}, what were you doing in {_room(room)}?", [eid, body["event_id"]], 3)
    for player, belief in beliefs.items():
        if player in partners or player == observation.agent_id:
            continue
        if belief["suspicion"] >= 0.5:
            add("ask", player, f"{player}, where were you, and who can vouch for you?", belief["evidence_ids"][:2], 4)
        elif not impostor and belief.get("trust", 0) >= 0.5:
            sightings = [m for m in recalled if m["kind"] == "TASK_SEEN" and m["details"].get("actor") == player]
            if sightings:
                add("tell", player, f"I saw {player} do a task in {_room(sightings[0]['details'].get('room'))}; {player} is fine by me.",
                    [sightings[0]["event_id"]], 5)
    return items
