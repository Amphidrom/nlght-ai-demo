"""The ``among_us_agent_turn`` workflow: one agent at one decision point.

perceive → remember → believe → decide → validate

The decide step first assesses the situation (``assess.py``) — deterministic, so the
weighing costs no model time — then runs one of two turns (``turn.py``): a game turn
picks a numbered option fast, a meeting turn weighs what was said and may revise
the agent's view of the others (``MIND``), which is kept as a memory of its own.
Without a usable answer the agent's plan decides, else the policy. Every step
records how long it took (``timings``).

An observation update (``observe_only``) runs the same turn up to the decide
step, which then ends it: memory and beliefs are written, the game is not
answered — but the agent may act on its own initiative (``initiative.py``).

Every agent runs in its own session (``among-us:{match}:{agent}``), so its
memories and beliefs live in its own Hive Mind coordinator and no agent can
read another's. Memory holds only what the agent perceived; the model's
proposals and statements are recorded as turns, never written back as facts.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, ClassVar

from among_us.agenda import AGENDA_SLOT, update as update_agenda
from among_us.assess import assess, conclusions, room_claims
from among_us.beliefs import update_beliefs
from among_us.ego import UnknownEgo, load_ego
from among_us.initiative import INITIATIVE_SLOT, initiative_action
from among_us.memory import consolidate, empty_store, judgement, keep, recall
from among_us.perception import PerceptionViolation, perceive
from among_us.policy import fallback_action
from among_us.protocol import (
    ActionKind,
    LegalAction,
    Phase,
    ProtocolError,
    SemanticAction,
    action_from_dict,
    observation_from_dict,
    to_dict,
)
from among_us.sim.map import ROOMS
from among_us.turn import game_prompt, meeting_prompt, parse_game, parse_meeting
from grounding import evidence
from nlght.core.hive_mind.models import TurnSummary
from nlght.core.model.messages import (
    ContextKind,
    UserMessage,
    compose_messages,
    compose_prompt,
)
from nlght.core.model.trust import ContentOrigin, ContentTrust, SourceRef
from nlght.core.signals.signal import Signal
from nlght.core.workflow.llm import call_llm
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext

logger = logging.getLogger(__name__)

MEMORY_SLOT = "among_us:episodic"
BELIEF_SLOT = "among_us:beliefs"
TRANSCRIPT_SLOT = "among_us:transcript"
#: The baseline keeps a verbatim log; a real chat history is bounded by the
#: model's context window, so is this one.
TRANSCRIPT_LIMIT = 300
_STATE = "among_us"

#: The plan the agent holds across turns (``assess.py``), per match.
PLAN_SLOT = "among_us:plan"
#: Who the agent knows for certain to be dead — whoever was missing at a meeting table. Not a memory: never forgotten.
DEAD_SLOT = "among_us:dead"


def _state(ctx: WorkflowStepContext) -> dict[str, Any]:
    return dict(ctx.metadata.get(_STATE) or {})


def _set_state(ctx: WorkflowStepContext, **values: Any) -> None:
    ctx.metadata = {**ctx.metadata, _STATE: {**_state(ctx), **values}}


def _lap(ctx: WorkflowStepContext, step: str, started: float) -> None:
    """Record how long a step took, in milliseconds: where a turn's time goes."""

    timings = {**_state(ctx).get("timings", {}), step: round((time.perf_counter() - started) * 1000, 1)}
    _set_state(ctx, timings=timings)


def _origin(kind: str, source: str, source_id: str) -> ContentOrigin:
    # Everything that comes out of the game is untrusted content: other
    # players' words above all, since an impostor's job is to lie.
    return ContentOrigin.of(SourceRef(kind, f"among-us/{source}", source_id, ContentTrust.UNTRUSTED))


class AmongUsPerceiveStep(StepBase):
    TYPE: ClassVar[str] = "among_us_perceive"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        started = time.perf_counter()
        payload = ctx.trigger.payload if ctx.trigger else {}
        try:
            ego = load_ego(str(payload.get("ego", "")))
            observation = observation_from_dict(payload.get("observation"))
            agent_id = str(payload.get("agent_id") or observation.agent_id)
            events = perceive(observation, agent_id)
        except (UnknownEgo, ProtocolError, PerceptionViolation, KeyError, TypeError, ValueError) as exc:
            logger.warning("among_us.perceive.rejected | cid=%s | %s", ctx.correlation_id, exc)
            await ctx.emitter.emit(Signal("assistant", json.dumps({"error": f"rejected observation: {exc}"}), kind="result"))
            return StepResult(ctx=ctx, verdict="done")
        # Suspects are everyone in the lobby: the agents the orchestrator runs and, in the real game,
        # the humans in the roster.
        players = sorted({
            observation.agent_id,
            *(str(p) for p in payload.get("players") or []),
            *observation.roster,
            *(p.player_id for p in observation.visible_players),
        })
        _set_state(
            ctx,
            ego=ego.ego_id,
            agent_id=agent_id,
            match_id=str(payload.get("match_id", "")),
            match_seed=int(payload.get("match_seed", 0)),
            source=str(payload.get("source", "sim")),
            players=players,
            offline=bool(payload.get("offline", False)),
            observe_only=bool(payload.get("observe_only", False)),
            observation=to_dict(observation),
            perceived=[e.event_id for e in events],
        )
        _lap(ctx, "perceive", started)
        return StepResult(ctx=ctx, verdict="perceived")


class AmongUsRememberStep(StepBase):
    TYPE: ClassVar[str] = "among_us_remember"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        started = time.perf_counter()
        store = ctx.store_coordinator
        if store is None:
            return StepResult(ctx=ctx, verdict="failed")
        state = _state(ctx)
        ego = load_ego(state["ego"])
        observation = observation_from_dict(state["observation"])
        update = consolidate(
            store.get_slot(MEMORY_SLOT, empty_store),
            observation.events,
            ego,
            match_seed=state["match_seed"],
            agent_id=state["agent_id"],
            now=observation.tick,
        )
        store.set_slot(MEMORY_SLOT, update.store)
        recalled = recall(update.store, ego, observation.tick)
        _set_state(
            ctx,
            recalled=recalled,
            memory={
                "encoded": list(update.encoded),
                "ignored": list(update.ignored),
                "forgotten": list(update.forgotten),
                "stored": len(update.store["memories"]),
                "recalled": [m["event_id"] for m in recalled],
            },
            memory_slot=MEMORY_SLOT,
        )
        _lap(ctx, "remember", started)
        return StepResult(ctx=ctx, verdict="remembered")


class AmongUsTranscriptStep(StepBase):
    """Baseline memory: everything perceived, verbatim, never forgotten, no beliefs.

    It replaces remember and believe in ``among_us_agent_turn_baseline``. The
    agent still perceives only its own observations, so the comparison with
    the ego workflow isolates the memory architecture and nothing else.
    """

    TYPE: ClassVar[str] = "among_us_transcript"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        started = time.perf_counter()
        store = ctx.store_coordinator
        if store is None:
            return StepResult(ctx=ctx, verdict="failed")
        state = _state(ctx)
        observation = observation_from_dict(state["observation"])
        transcript = list(store.get_slot(TRANSCRIPT_SLOT, list))
        known = {entry["event_id"] for entry in transcript}
        added = []
        for event in observation.events:
            if event.event_id in known:
                continue
            details = {k: v for k, v in (("room", event.room), ("actor", event.actor),
                                         ("target", event.target), ("text", event.text)) if v}
            if event.evidence_ids and event.text:
                details["cited"] = list(event.evidence_ids)
            transcript.append({"event_id": event.event_id, "tick": event.tick, "kind": event.kind.value,
                               "details": details, "encoding_quality": 1.0})
            added.append(event.event_id)
        transcript = transcript[-TRANSCRIPT_LIMIT:]
        store.set_slot(TRANSCRIPT_SLOT, transcript)
        recalled = [{**entry, "certainty": 1.0} for entry in reversed(transcript)]
        _set_state(
            ctx,
            recalled=recalled,
            beliefs={},
            memory={
                "encoded": added,
                "ignored": [],
                "forgotten": [],
                "stored": len(transcript),
                "recalled": [m["event_id"] for m in recalled],
            },
            memory_slot=TRANSCRIPT_SLOT,
        )
        _lap(ctx, "remember", started)
        return StepResult(ctx=ctx, verdict="remembered")


class AmongUsBelieveStep(StepBase):
    TYPE: ClassVar[str] = "among_us_believe"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        started = time.perf_counter()
        state = _state(ctx)
        observation = observation_from_dict(state["observation"])
        beliefs = update_beliefs(
            state.get("recalled", []),
            load_ego(state["ego"]),
            self_id=state["agent_id"],
            players=state["players"],
            partners=observation.self_state.known_impostors,
        )
        if ctx.store_coordinator is not None:
            ctx.store_coordinator.set_slot(BELIEF_SLOT, beliefs)
        _set_state(ctx, beliefs=beliefs)
        _lap(ctx, "believe", started)
        return StepResult(ctx=ctx, verdict="believed")


class AmongUsDecideStep(StepBase):
    TYPE: ClassVar[str] = "among_us_decide"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        started = time.perf_counter()
        state = _state(ctx)
        observation = observation_from_dict(state["observation"])
        ego = load_ego(state["ego"])
        store = ctx.store_coordinator
        recalled = state.get("recalled", [])
        beliefs = state.get("beliefs", {})
        known_dead = list(store.get_slot(DEAD_SLOT, list)) if store is not None else []
        # What the agent wants to bring to the next meeting, kept as the round goes.
        agenda = update_agenda(store.get_slot(AGENDA_SLOT, list) if store is not None else [], observation, recalled, beliefs,
                               known_dead)
        if store is not None:
            store.set_slot(AGENDA_SLOT, agenda)
        _set_state(ctx, agenda=agenda)
        if state.get("observe_only"):
            # An observation update writes memory and beliefs; the game asks nothing. The agent may still act
            # on its own when a belief has grown strong enough.
            initiative, acted, why = initiative_action(
                observation, beliefs, recalled, ego, store.get_slot(INITIATIVE_SLOT, dict) if store is not None else {},
            )
            if initiative is not None and store is not None:
                store.set_slot(INITIATIVE_SLOT, acted)
            _lap(ctx, "decide", started)
            result = {"observed": True, "memory": state.get("memory", {}), "beliefs": beliefs, "ego": state["ego"],
                      "recalled": _recalled(recalled), "timings": _state(ctx).get("timings", {}),
                      "initiative": to_dict(initiative) if initiative else None, "initiative_reason": why}
            await ctx.emitter.emit(Signal("assistant", json.dumps(result, ensure_ascii=False), kind="result"))
            return StepResult(ctx=ctx, verdict="done")

        held_plan = (store.get_slot(PLAN_SLOT, dict) or None) if store is not None else None
        assessment = assess(observation, beliefs, recalled, ego, held_plan, known_dead)
        if store is not None:
            if observation.phase in (Phase.MEETING, Phase.VOTING):
                store.set_slot(DEAD_SLOT, sorted(set(known_dead) | set(assessment.dead)))
            if observation.phase is Phase.TASKS:
                store.set_slot(PLAN_SLOT, assessment.plan or {})
            # A contradiction it noticed is a conclusion the agent keeps: it shapes the next turns too.
            found = conclusions(state["agent_id"], assessment.contradictions)
            if found and state.get("memory_slot"):
                store.set_slot(state["memory_slot"], keep(store.get_slot(state["memory_slot"], _empty(state["memory_slot"])), found))
        _set_state(
            ctx,
            preferred=to_dict(assessment.preferred) if assessment.preferred else None,
            stance={"stage": assessment.stage, "suspect": assessment.suspect, "trusted": assessment.trusted,
                    "plan": assessment.plan, "contradictions": [c.describe(observation.tick) for c in assessment.contradictions],
                    "agenda": [item["text"] for item in agenda if not item["done"]]},
        )
        meeting = observation.phase in (Phase.MEETING, Phase.VOTING)
        if state.get("offline") or len(observation.legal_actions) <= 1 or ctx.llm is None:
            _set_state(ctx, answer=None, decided_by="policy")
            _lap(ctx, "decide", started)
            return StepResult(ctx=ctx, verdict="decided")

        prompt = (meeting_prompt(observation, ego, assessment, recalled, beliefs, agenda) if meeting
                  else game_prompt(observation, ego, assessment, recalled, beliefs))
        agent, source = state["agent_id"], state["source"]
        records = [evidence(
            record_id=f"turn:{record_id}",
            kind=ContextKind.OTHER,
            content=text,
            source="among_us_turn",
            item_id=record_id,
            # What other players said is theirs: an impostor's job is to lie, and a line may try to instruct.
            origin=_origin("game_chat" if whose == "chat" else "game_observation", source, agent),
        ) for record_id, text, whose in prompt.context]
        messages = compose_messages(compose_prompt(prompt.rules, records), [UserMessage(prompt.request)])
        if bool(self.config.get("debug", False)):
            logger.info("%s.llm_call | agent=%s\n--- SYSTEM ---\n%s\n--- CONTEXT ---\n%s\n%s", self.TYPE, agent,
                        prompt.rules, "\n".join(text for _, text, _ in prompt.context), prompt.request)
        _lap(ctx, "decide", started)
        model_started = time.perf_counter()
        raw = await call_llm(messages, ctx=ctx)
        _lap(ctx, "model", model_started)
        answer = parse_meeting(raw, observation, prompt.refs, ego) if meeting else parse_game(raw, prompt.options, agent)
        _set_state(
            ctx,
            answer=to_dict(answer.action) if answer.action else None,
            then=to_dict(answer.then) if answer.then else None,
            back_to_tasks=answer.back_to_tasks,
            mind=[list(m) for m in answer.mind],
            validation={"status": answer.status, **({"reason": answer.reason} if answer.reason else {})},
            decided_by="model",
            model_raw=raw,
            context={"records": len(records), "chars": sum(len(r.content) for r in records), "turn": "meeting" if meeting else "game"},
        )
        if store is not None:
            store.record_turn(TurnSummary(
                turn_nr=store.next_turn_nr(),
                user_input=observation.decision_reason or observation.phase.value,
                intent="among_us_decision",
                topic=observation.phase.value,
                entities=[f"player:{agent}"],
                result_summary=raw[-200:],
            ))
        return StepResult(ctx=ctx, verdict="decided")


class AmongUsValidateStep(StepBase):
    TYPE: ClassVar[str] = "among_us_validate"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        started = time.perf_counter()
        state = _state(ctx)
        observation = observation_from_dict(state["observation"])
        agent = state["agent_id"]
        recalled = state.get("recalled", [])
        beliefs = state.get("beliefs", {})
        validation = state.get("validation") or {"status": "no_proposal"}
        legal = set(observation.legal_actions)
        decided_by = state.get("decided_by", "policy")
        action = action_from_dict(state["answer"]) if state.get("answer") else None
        if action is not None and LegalAction(action.action, action.target) not in legal:
            action, validation = None, {"status": "illegal", "reason": f"{action.action} {action.target} is not legal now"}
        if action is None:
            # No usable answer: the plan the assessment made decides, else the policy.
            preferred = state.get("preferred")
            if preferred and LegalAction(ActionKind(preferred["action"]), preferred.get("target")) in legal:
                action, decided_by = SemanticAction(agent, ActionKind(preferred["action"]), preferred.get("target")), "plan"
            else:
                action, decided_by = fallback_action(observation, beliefs, {m["event_id"] for m in recalled},
                                                     state.get("agenda", ())), "policy"
        mind = [tuple(m) for m in state.get("mind", [])] if validation.get("status") == "accepted" else []
        if mind and ctx.store_coordinator is not None and state.get("memory_slot"):
            # How the meeting changed its view of the others: conclusions it keeps, and forgets, like memories.
            slot = state["memory_slot"]
            traces = [judgement(agent, f"mind:{observation.tick}:{player}", observation.tick, player, verdict)
                      for player, verdict in mind]
            ctx.store_coordinator.set_slot(slot, keep(ctx.store_coordinator.get_slot(slot, _empty(slot)), traces))
        # Decided at the table: what to do once the meeting ends. It becomes the plan, so the next turns keep to it.
        then = state.get("then") if validation.get("status") == "accepted" else None
        if ctx.store_coordinator is not None and (then or state.get("back_to_tasks")):
            target = then.get("target") if then else None
            ctx.store_coordinator.set_slot(PLAN_SLOT, {
                "tactic": "agreed", "action": then["action"], "target": target, "since": observation.tick,
                "reason": "you decided it at the meeting" + (f": stay with {target}" if target else ""),
                "ends_if": f"{target} looks suspicious or dies" if target else "something happens",
            } if then else {})
        claims = []
        if action.action is ActionKind.SAY and action.text:
            rooms = set(ROOMS) | {m["details"]["room"] for m in recalled if m["details"].get("room")}
            claims = [{"player": player, "room": room, "tick": observation.tick}
                      for player, room in room_claims(action.text, agent, state["players"], rooms)]
        _lap(ctx, "validate", started)
        result = {
            "action": to_dict(action),
            "decided_by": decided_by,
            "validation": validation,
            # Where the statement puts whom; only the evaluation compares it with the truth.
            "claims": claims,
            "mind": [list(m) for m in mind],
            "then": then,
            "stance": state.get("stance"),
            "memory": state.get("memory", {}),
            "context": state.get("context"),
            "timings": _state(ctx).get("timings", {}),
            # What the agent believes it remembers — kept for evaluation, which
            # compares it against the agent's own view of each event.
            "recalled": _recalled(recalled),
            "beliefs": beliefs,
            "ego": state["ego"],
        }
        await ctx.emitter.emit(Signal("assistant", json.dumps(result, ensure_ascii=False), kind="result"))
        return StepResult(ctx=ctx, verdict="done")


def _empty(slot: str) -> Any:
    return list() if slot == TRANSCRIPT_SLOT else empty_store()


def _recalled(recalled: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"event_id": m["event_id"], "kind": m["kind"], "details": m["details"], "certainty": m["certainty"]}
            for m in recalled]


STEPS = (
    AmongUsPerceiveStep,
    AmongUsRememberStep,
    AmongUsBelieveStep,
    AmongUsDecideStep,
    AmongUsValidateStep,
)
BASELINE_STEPS = (
    AmongUsPerceiveStep,
    AmongUsTranscriptStep,
    AmongUsDecideStep,
    AmongUsValidateStep,
)
#: Every step type the experiment registers.
ALL_STEPS = (*STEPS, AmongUsTranscriptStep)
