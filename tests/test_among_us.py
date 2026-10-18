from __future__ import annotations

import asyncio
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from among_us.adapter import _match_options
from among_us.beliefs import update_beliefs
from among_us.ego import ego_ids, load_ego
from among_us.environment import BridgeEnvironment, SimEnvironment
from among_us.evaluation import differentiation_probe, evaluate
from among_us.match import MatchRecord, PipelineAgentRunner, run_match, session_key
from among_us.memory import consolidate, empty_store, recall
from among_us.perception import PerceptionViolation, perceive
from among_us.policy import fallback_action
from among_us.protocol import (
    ActionKind,
    ActionResult,
    DecisionRequest,
    EventKind,
    Hello,
    LegalAction,
    MatchEnd,
    Observation,
    ObservedEvent,
    Phase,
    ProtocolError,
    Role,
    SelfState,
    SemanticAction,
    decode,
    encode,
    to_dict,
)
from among_us.sim.engine import SimEngine
from among_us.steps import validate_proposal
from fastapi import HTTPException
from nlght.core.model.messages import TrustedInstructionMessage, UntrustedContextMessage
from nlght.core.model.trust import ContentTrust
from nlght.core.signals.signal import Signal

EGOS = ["archive", "bubbles", "echo", "maverick", "sherlock"]
FIXTURES = Path("among_us/protocol_fixtures")


class FakeBackend:
    """A provider backend for ``call_llm``: answers from a script, records prompts."""

    def __init__(self, answer) -> None:
        self.answer = answer
        self.calls: list[list] = []

    def bind(self, *, model, emitter, stream, **_):
        backend = self

        class Bound:
            async def call(self, messages) -> None:
                backend.calls.append(list(messages))
                await emitter.emit(Signal("assistant", backend.answer(messages), kind="result"))

        return Bound()


def fake_llm(answer) -> SimpleNamespace:
    backend = FakeBackend(answer)
    return SimpleNamespace(_backend=backend, _model="fake", token_budget=None)


def first_legal_answer(messages) -> str:
    """Pick the last legal action from the observation record, citing nothing."""
    for message in messages:
        if isinstance(message, UntrustedContextMessage):
            for record in message.records:
                if record.record_id == "observation:self":
                    legal = json.loads(record.content)["legal_actions"]
                    choice = legal[-1]
                    text = "I was doing my tasks." if choice["action"] == "SAY" else None
                    return json.dumps({"action": choice["action"], "target": choice["target"], "text": text,
                                       "goal": "TEST", "evidence_ids": []})
    return "no idea"


def play(seed: int, runner=None, *, offline: bool = True) -> tuple[MatchRecord, SimEnvironment]:
    environment = SimEnvironment(EGOS, seed=seed)
    record = MatchRecord(f"m{seed}", seed, {e: e for e in EGOS}, "sim")
    asyncio.run(run_match(environment, runner or PipelineAgentRunner(), record, offline=offline))
    return record, environment


def crew_observation(**overrides) -> Observation:
    values = dict(
        agent_id="sherlock",
        tick=10,
        phase=Phase.MEETING,
        self_state=SelfState("sherlock", "cafeteria", Role.CREWMATE),
        legal_actions=(LegalAction(ActionKind.SAY), LegalAction(ActionKind.WAIT)),
    )
    values.update(overrides)
    return Observation(**values)


class ProtocolTests(unittest.TestCase):
    def test_every_fixture_round_trips(self) -> None:
        for name in ("hello", "decision_request", "action", "action_result", "match_end"):
            raw = (FIXTURES / f"{name}.json").read_text(encoding="utf-8")
            message = decode(raw)
            self.assertEqual(json.loads(raw), json.loads(encode(message)), name)

    def test_an_observation_that_discloses_a_role_is_refused(self) -> None:
        raw = (FIXTURES / "invalid_leaky_observation.json").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "unknown fields"):
            decode(raw)

    def test_version_and_type_are_checked(self) -> None:
        with self.assertRaises(ProtocolError):
            decode(json.dumps({"type": "hello", "protocol": 2, "game_version": "x", "source": "y"}))
        with self.assertRaises(ProtocolError):
            decode(json.dumps({"type": "teleport", "protocol": 1}))


class SimulationTests(unittest.TestCase):
    def test_same_seed_replays_identically(self) -> None:
        first, env_a = play(4)
        second, env_b = play(4)
        self.assertEqual(env_a.ground_truth().events, env_b.ground_truth().events)
        strip = lambda r: [(e["agent_id"], e["tick"], e["action"], e["result"].get("recalled")) for e in r.entries if e["type"] == "decision"]
        self.assertEqual(strip(first), strip(second))
        self.assertNotEqual(env_a.ground_truth().events, play(5)[1].ground_truth().events)

    def test_matches_end_with_a_winner_and_only_legal_actions(self) -> None:
        for seed in range(6):
            record, _ = play(seed)
            self.assertEqual("finished", record.status, record.error)
            self.assertIn(record.winner["winner"], {"crew", "impostors"})
            self.assertTrue(all(e["accepted"] for e in record.entries if e["type"] == "decision"))

    def test_impostors_win_at_parity(self) -> None:
        engine = SimEngine(["a", "b", "c", "d"], seed=1, kill_cooldown=0)
        impostor = next(pid for pid in engine.player_ids if engine.role_of(pid) is Role.IMPOSTOR)
        crew = [pid for pid in engine.player_ids if pid != impostor]
        for victim in crew[:2]:
            for pid in engine.alive():
                engine._players[pid].room = "electrical" if pid in (impostor, victim) else "admin"
            self.assertTrue(engine.submit(SemanticAction(impostor, ActionKind.KILL, victim)).accepted)
            end = engine.advance()
        self.assertEqual("impostors", end.winner)
        self.assertEqual(2, sum(1 for e in engine.events if e.kind == "KILL"))

    def test_observations_contain_only_the_agents_own_view(self) -> None:
        engine = SimEngine(EGOS, seed=2)
        delivered: dict[str, set[str]] = {pid: set() for pid in EGOS}
        while engine.winner is None:
            for pid in engine.alive():
                if engine.needs_decision(pid):
                    observation = engine.observe(pid)
                    perceive(observation, pid)
                    delivered[pid] |= {e.event_id for e in observation.events}
                    for player in observation.visible_players:
                        self.assertFalse(hasattr(player, "role"))
                    if observation.self_state.role is Role.CREWMATE:
                        self.assertEqual((), observation.self_state.known_impostors)
                    engine.submit(fallback_action(observation, {}, set()))
            engine.advance()
        for pid, ids in delivered.items():
            witnessed = {event.event_id for event in engine.events if pid in event.witnesses}
            self.assertLessEqual(ids, witnessed, pid)

    def test_a_fake_task_looks_like_a_real_one(self) -> None:
        _, environment = play(1)
        fakes = [e for e in environment.ground_truth().events if e["kind"] == "FAKE_TASK"]
        for event in fakes:
            for view in event["views"].values():
                self.assertEqual(EventKind.TASK_SEEN.value, view["kind"])


class PerceptionTests(unittest.TestCase):
    def test_observation_for_another_agent_is_refused(self) -> None:
        with self.assertRaises(PerceptionViolation):
            perceive(crew_observation(), "bubbles")

    def test_crewmate_never_learns_impostors(self) -> None:
        leaked = crew_observation(self_state=SelfState("sherlock", "cafeteria", Role.CREWMATE, known_impostors=("echo",)))
        with self.assertRaises(PerceptionViolation):
            perceive(leaked, "sherlock")


class MemoryTests(unittest.TestCase):
    def _stream(self) -> list[ObservedEvent]:
        events = []
        for tick in range(1, 31):
            actor = EGOS[tick % 5]
            events.append(ObservedEvent(f"e{tick:04d}", tick, EventKind.PLAYER_ENTERED, "admin", actor))
            events.append(ObservedEvent(f"t{tick:04d}", tick, EventKind.TASK_SEEN, "admin", actor, "swipe_card"))
        events.append(ObservedEvent("c0001", 30, EventKind.CHAT, None, "maverick", text="I saw echo near admin, vote echo"))
        return events

    def test_three_egos_remember_the_same_events_differently(self) -> None:
        probe = differentiation_probe(self._stream(), ["sherlock", "archive", "bubbles"], match_seed=11, now=30)
        for pair, distance in probe["recall_distance"].items():
            self.assertGreater(distance, 0.0, pair)
        recalled = {ego: len(m["recalled"]) for ego, m in probe["memories"].items()}
        self.assertGreater(recalled["archive"], recalled["bubbles"])
        self.assertGreater(recalled["sherlock"], recalled["bubbles"])
        self.assertEqual(probe, differentiation_probe(self._stream(), ["sherlock", "archive", "bubbles"], match_seed=11, now=30))

    def test_forgotten_events_never_resurface(self) -> None:
        bubbles = load_ego("bubbles")
        stream = self._stream()
        update = consolidate(empty_store(), stream[:10], bubbles, match_seed=3, agent_id="bubbles", now=5)
        early = set(update.store["memories"])
        later = consolidate(update.store, stream, bubbles, match_seed=3, agent_id="bubbles", now=120)
        self.assertTrue(early)
        self.assertTrue(early & set(later.forgotten))
        again = consolidate(later.store, stream, bubbles, match_seed=3, agent_id="bubbles", now=121)
        self.assertFalse(set(later.forgotten) & set(again.store["memories"]))
        self.assertEqual((), again.encoded)

    def test_archive_does_not_distort_details(self) -> None:
        archive = load_ego("archive")
        update = consolidate(empty_store(), self._stream(), archive, match_seed=9, agent_id="archive", now=30)
        for memory in recall(update.store, archive, 30):
            if "room" in memory["details"]:
                self.assertEqual("admin", memory["details"]["room"])

    def test_witnessed_kill_makes_the_killer_the_prime_suspect(self) -> None:
        kill = ObservedEvent("k1", 5, EventKind.KILL_WITNESSED, "electrical", "echo", "bubbles")
        update = consolidate(empty_store(), [kill], load_ego("bubbles"), match_seed=1, agent_id="sherlock", now=6)
        beliefs = update_beliefs(recall(update.store, load_ego("sherlock"), 6), load_ego("sherlock"),
                                 self_id="sherlock", players=EGOS)
        self.assertEqual(["k1"], beliefs["echo"]["evidence_ids"])
        self.assertEqual(max(b["suspicion"] for b in beliefs.values()), beliefs["echo"]["suspicion"])

    def test_every_profile_loads(self) -> None:
        self.assertEqual(sorted(EGOS), list(ego_ids()))
        for ego in ego_ids():
            self.assertTrue(load_ego(ego).persona)


class ValidationTests(unittest.TestCase):
    def test_accepts_a_legal_supported_statement(self) -> None:
        action, status = validate_proposal(
            {"action": "SAY", "text": "I saw echo leave electrical.", "evidence_ids": ["e1"]},
            crew_observation(), {"e1"}, load_ego("sherlock"),
        )
        self.assertEqual("accepted", status["status"])
        self.assertEqual(("e1",), action.evidence_ids)

    def test_refuses_evidence_the_agent_cannot_recall(self) -> None:
        action, status = validate_proposal(
            {"action": "SAY", "text": "echo did it", "evidence_ids": ["e9"]},
            crew_observation(), {"e1"}, load_ego("sherlock"),
        )
        self.assertIsNone(action)
        self.assertEqual("unsupported", status["status"])

    def test_refuses_an_illegal_action(self) -> None:
        action, status = validate_proposal({"action": "KILL", "target": "echo"}, crew_observation(), set(), load_ego("sherlock"))
        self.assertIsNone(action)
        self.assertEqual("illegal", status["status"])

    def test_truncates_long_statements_to_the_ego(self) -> None:
        action, _ = validate_proposal({"action": "SAY", "text": "word " * 100}, crew_observation(), set(), load_ego("bubbles"))
        self.assertLessEqual(len(action.text.split()), load_ego("bubbles").max_words + 1)


class AgentTurnTests(unittest.TestCase):
    def test_model_sees_memories_only_as_untrusted_context(self) -> None:
        llm = fake_llm(first_legal_answer)
        record, _ = play(3, PipelineAgentRunner(llm), offline=False)
        calls = llm._backend.calls
        self.assertTrue(calls)
        for messages in calls[:20]:
            trusted = [m for m in messages if isinstance(m, TrustedInstructionMessage)]
            self.assertEqual(1, len(trusted))
            self.assertNotIn("event_id\":", trusted[0].content)
            for message in messages:
                if isinstance(message, UntrustedContextMessage):
                    for item in message.records:
                        self.assertIs(ContentTrust.UNTRUSTED, item.provenance.origin.trust)
        chat_origins = {
            ref.kind
            for messages in calls for m in messages if isinstance(m, UntrustedContextMessage)
            for item in m.records if item.record_id.startswith("memory:") and '"CHAT"' in item.content
            for ref in item.provenance.origin.sources
        }
        if chat_origins:
            self.assertEqual({"game_chat"}, chat_origins)
        self.assertEqual("finished", record.status, record.error)

    def test_garbage_from_the_model_falls_back_to_a_legal_action(self) -> None:
        record, _ = play(2, PipelineAgentRunner(fake_llm(lambda _: '{"action":"TELEPORT"}')), offline=False)
        self.assertEqual("finished", record.status, record.error)
        decisions = [e for e in record.entries if e["type"] == "decision"]
        self.assertTrue(all(d["accepted"] for d in decisions))
        self.assertTrue(any(d["result"]["validation"]["status"] == "malformed" for d in decisions))

    def test_each_agent_keeps_its_own_session(self) -> None:
        runner = PipelineAgentRunner()
        record, _ = play(1, runner)
        self.assertEqual({session_key(record.match_id, e) for e in EGOS} >= set(runner.sessions), True)
        stores = {key: s.slots.get("among_us:episodic", {}).get("memories", {}) for key, s in runner.sessions.items()}
        self.assertGreater(len({json.dumps(m, sort_keys=True) for m in stores.values()}), 1)


class EvaluationTests(unittest.TestCase):
    def test_a_simulated_match_has_no_memory_leakage(self) -> None:
        for seed in (0, 3, 7):
            record, _ = play(seed)
            report = evaluate(record)
            self.assertEqual(0, report["memory_leakage"]["violations"], report["memory_leakage"]["examples"])
            self.assertGreater(report["memory_leakage"]["references"], 0)
            self.assertEqual(1.0, report["action_validity"]["game_acceptance_rate"])

    def test_recall_differs_by_ego(self) -> None:
        record, _ = play(0)
        recall_report = evaluate(record)["recall"]
        coverage = {r["ego"]: r["coverage"] for r in recall_report.values()}
        self.assertEqual(set(EGOS), set(coverage))
        self.assertGreater(len(set(coverage.values())), 1)


class BridgeEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_speaks_protocol_v1_over_ndjson(self) -> None:
        received: list[str] = []
        observation = decode((FIXTURES / "decision_request.json").read_text(encoding="utf-8")).observation

        async def bridge(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            writer.write((encode(Hello("2026.3.31", "test-bridge", ("sherlock",))) + "\n").encode())
            writer.write(b'{"type":"decision_request","protocol":1,"observation":{"agent_id":"x","role":"IMPOSTOR"}}\n')
            writer.write((encode(DecisionRequest(observation)) + "\n").encode())
            await writer.drain()
            line = await reader.readline()
            received.append(line.decode())
            writer.write((encode(ActionResult("sherlock", True)) + "\n").encode())
            writer.write((encode(MatchEnd("crew", "test")) + "\n").encode())
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(bridge, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server:
            environment = BridgeEnvironment("127.0.0.1", port, decision_wait=2)
            hello = await environment.connect()
            self.assertEqual(("sherlock",), hello.players)
            decisions = await environment.next_decisions()
            self.assertEqual(["sherlock"], [o.agent_id for o in decisions])
            result = await environment.submit(SemanticAction("sherlock", ActionKind.WAIT))
            self.assertTrue(result.accepted)
            self.assertEqual("action", json.loads(received[0])["type"])
            for _ in range(20):
                if await environment.advance():
                    break
                await asyncio.sleep(0.05)
            self.assertEqual("crew", (await environment.advance()).winner)
            await environment.close()


class WorkflowSeedTests(unittest.TestCase):
    def _seed(self) -> str:
        return Path("seed/08_among_us.sql").read_text(encoding="utf-8")

    def test_every_seeded_step_type_is_registered(self) -> None:
        import main  # noqa: F401
        from nlght.adapters.outbound.workflow.registry import step_registry

        seeded = set(re.findall(r"'(among_us_[a-z]+)',\n", self._seed()))
        self.assertEqual(5, len(seeded))
        self.assertEqual(set(), seeded - set(getattr(step_registry, "_registry")))

    def test_only_the_decide_step_selects_a_model(self) -> None:
        seed = self._seed()
        self.assertEqual(1, seed.count('"model_provider":"ollama"'))
        self.assertIn("'decide', 'among_us_decide'", seed)

    def test_transitions_point_at_steps_of_this_workflow(self) -> None:
        seed = self._seed()
        defined = set(re.findall(r"\('(a8100000-0000-0000-0000-0000000000\d\d)', 'a8100000-0000-0000-0000-000000000002'", seed))
        targets = set(re.findall(r'":"(a8100000-[0-9a-f-]+)"', seed))
        self.assertTrue(targets)
        self.assertEqual(set(), targets - defined)

    def test_base_seed_does_not_load_the_experiment(self) -> None:
        self.assertNotIn("08_among_us", Path("seed.sql").read_text(encoding="utf-8").replace("-- ", ""))


class AdapterOptionTests(unittest.TestCase):
    def test_bridge_port_is_restricted_to_the_bridge_range(self) -> None:
        with self.assertRaises(HTTPException):
            _match_options({"environment": "bridge", "players": {"p1": "sherlock"}, "bridge_port": 5432})
        options = _match_options({"environment": "bridge", "players": {"p1": "sherlock"}})
        self.assertEqual("127.0.0.1", options["bridge_host"])

    def test_simulated_match_needs_four_or_five_known_egos(self) -> None:
        with self.assertRaises(HTTPException):
            _match_options({"egos": ["sherlock", "archive", "bubbles"]})
        with self.assertRaises(HTTPException):
            _match_options({"egos": ["sherlock", "archive", "bubbles", "hacker"]})
        self.assertEqual(5, len(_match_options({})["assignment"]))

    def test_action_payload_is_plain_json(self) -> None:
        self.assertEqual("MOVE_TO", to_dict(SemanticAction("a", ActionKind.MOVE_TO, "admin"))["action"])
