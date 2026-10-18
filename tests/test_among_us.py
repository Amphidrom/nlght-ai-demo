from __future__ import annotations

import asyncio
from collections import Counter
import json
import tomllib
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from among_us.adapter import _match_options
from among_us.beliefs import update_beliefs
from among_us.ego import InvalidEgo, ego_ids, file_schema, load_ego, parse_ego
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
from among_us.scheduler import DecisionScheduler, SchedulerConfig
from among_us.sim.engine import MAX_PLAYERS, SimEngine, max_impostors
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

    def test_full_lobby_with_three_impostors_plays_to_the_end(self) -> None:
        egos = list(ego_ids())
        environment = SimEnvironment(egos, seed=21, impostors=3)
        record = MatchRecord("full", 21, {e: e for e in egos}, "sim")
        asyncio.run(run_match(environment, PipelineAgentRunner(), record, offline=True))
        self.assertEqual("finished", record.status, record.error)
        truth = environment.ground_truth()
        self.assertEqual(3, sum(1 for role in truth.roles.values() if role == "IMPOSTOR"))
        self.assertEqual(0, evaluate(record)["memory_leakage"]["violations"])

    def test_lobby_limits_follow_the_game(self) -> None:
        with self.assertRaises(ValueError):
            SimEngine([f"p{i}" for i in range(16)], seed=1)
        with self.assertRaises(ValueError):
            SimEngine([f"p{i}" for i in range(6)], seed=1, impostors=2)
        self.assertEqual((1, 2, 3), (max_impostors(6), max_impostors(8), max_impostors(15)))

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

    def test_meetings_do_not_reveal_where_the_others_were(self) -> None:
        engine = SimEngine(EGOS, seed=1)
        for pid in EGOS:
            engine._players[pid].room = "electrical" if pid == "echo" else "admin"
        engine.submit(SemanticAction("sherlock", ActionKind.WAIT))
        engine._start_meeting(engine._players["sherlock"], None, "test")
        observation = engine.observe("sherlock")
        self.assertEqual(Phase.MEETING, observation.phase)
        self.assertEqual({"meeting"}, {p.room for p in observation.visible_players})
        self.assertEqual("admin", observation.self_state.room)

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

    def test_a_full_lobby_has_an_ego_per_seat(self) -> None:
        self.assertEqual(MAX_PLAYERS, len(ego_ids()))
        self.assertLessEqual(set(EGOS), set(ego_ids()))
        profiles = [load_ego(ego) for ego in ego_ids()]
        for profile in profiles:
            self.assertTrue(profile.persona)
        # No two egos are cognitively identical.
        cognition = {p.model_dump_json(include={"attention", "encoding", "retention", "judgement"}) for p in profiles}
        self.assertEqual(len(profiles), len(cognition))

    def test_committed_schema_matches_the_model(self) -> None:
        committed = json.loads(Path("among_us/egos/ego.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(file_schema(), committed, "regenerate with: python -m among_us.ego")

    def test_ego_files_are_validated_strictly(self) -> None:
        valid = Path("among_us/egos/sherlock.toml").read_text(encoding="utf-8")
        broken = {
            "string number": valid.replace("fidelity = 0.95", 'fidelity = "0.95"'),
            "out of range": valid.replace("detail_noise = 0.02", "detail_noise = 1.5"),
            "unknown key": valid.replace("[voice]", "[voice]\nshout = true"),
            "unknown event kind": valid.replace("[attention.by_kind]", "[attention.by_kind]\nTELEPATHY = 1.0"),
            "id in file": "ego_id = \"other\"\n" + valid,
            "missing section": valid.split("[judgement]")[0],
        }
        for label, text in broken.items():
            with self.subTest(label), self.assertRaises(InvalidEgo):
                parse_ego("sherlock", text)
        self.assertEqual(load_ego("sherlock"), parse_ego("sherlock", valid))


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
        self.assertLessEqual(len(action.text.split()), load_ego("bubbles").voice.max_words + 1)


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
        self.assertEqual(6, len(seeded))
        self.assertEqual(set(), seeded - set(getattr(step_registry, "_registry")))

    def test_only_the_decide_steps_select_a_model(self) -> None:
        seed = self._seed()
        self.assertEqual(2, seed.count('"model_provider":"ollama"'))
        self.assertEqual(2, seed.count("'decide', 'among_us_decide'"))

    def test_baseline_differs_from_the_ego_workflow_only_in_memory(self) -> None:
        seed = self._seed()
        ego, baseline = seed.split("among_us_agent_turn_baseline", 1)
        ego_steps = re.findall(r"'(among_us_[a-z]+)',\n", ego)
        baseline_steps = re.findall(r"'(among_us_[a-z]+)',\n", baseline)
        self.assertEqual(["among_us_perceive", "among_us_remember", "among_us_believe", "among_us_decide", "among_us_validate"], ego_steps)
        self.assertEqual(["among_us_perceive", "among_us_transcript", "among_us_decide", "among_us_validate"], baseline_steps)

    def test_transitions_point_at_steps_of_this_workflow(self) -> None:
        seed = self._seed()
        defined = set(re.findall(r"\('(a8100000-0000-0000-0000-0000000000\d\d)', 'a8100000-0000-0000-0000-0000000000(?:02|22)'", seed))
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

    def test_simulated_match_needs_four_to_fifteen_known_egos(self) -> None:
        with self.assertRaises(HTTPException):
            _match_options({"egos": ["sherlock", "archive", "bubbles"]})
        with self.assertRaises(HTTPException):
            _match_options({"egos": ["sherlock", "archive", "bubbles", "hacker"]})
        with self.assertRaises(HTTPException):
            _match_options({"egos": ["sherlock", "archive", "bubbles", "echo"], "impostors": 2})
        options = _match_options({})
        self.assertEqual(MAX_PLAYERS, len(options["assignment"]))
        self.assertEqual(2, options["impostors"])

    def test_action_payload_is_plain_json(self) -> None:
        self.assertEqual("MOVE_TO", to_dict(SemanticAction("a", ActionKind.MOVE_TO, "admin"))["action"])


def task_observation(agent_id: str, reason: str = "goal reached", tick: int = 3, phase: Phase = Phase.TASKS) -> Observation:
    return Observation(
        agent_id=agent_id, tick=tick, phase=phase,
        self_state=SelfState(agent_id, "admin", Role.CREWMATE),
        legal_actions=(LegalAction(ActionKind.WAIT),), decision_reason=reason,
    )


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_meetings_and_salient_observations_go_first(self) -> None:
        batch = [
            task_observation("a"),
            task_observation("b", "opportunity"),
            task_observation("c", "observed: KILL_WITNESSED"),
            task_observation("d", phase=Phase.MEETING),
        ]
        order = [obs.agent_id for obs, _ in DecisionScheduler().plan(batch)]
        self.assertEqual(["d", "c", "b", "a"], order)

    async def test_budget_limits_model_turns_per_tick_but_not_meetings(self) -> None:
        scheduler = DecisionScheduler(SchedulerConfig(model_turns_per_tick=2))
        batch = [task_observation(f"p{i}") for i in range(5)] + [task_observation("m", phase=Phase.MEETING)]
        modes = dict((obs.agent_id, mode) for obs, mode in scheduler.plan(batch))
        self.assertEqual("model", modes["m"])
        self.assertEqual(2, sum(1 for agent, mode in modes.items() if agent != "m" and mode == "model"))
        self.assertEqual(3, sum(1 for mode in modes.values() if mode == "policy:budget"))
        # The budget is per tick: the next tick starts fresh.
        later = scheduler.plan([task_observation("p0", tick=4)])
        self.assertEqual("model", later[0][1])

    async def test_concurrency_is_capped(self) -> None:
        scheduler = DecisionScheduler(SchedulerConfig(max_concurrent_model_turns=3, model_turns_per_tick=15))
        running = peak = 0

        async def turn(observation: Observation, use_model: bool) -> dict:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            return {"use_model": use_model}

        results = await scheduler.run([task_observation(f"p{i}") for i in range(15)], turn)
        self.assertEqual(3, peak)
        self.assertTrue(all(result["use_model"] for _, result, _ in results))

    async def test_a_slow_model_turn_is_replaced_by_the_policy(self) -> None:
        scheduler = DecisionScheduler(SchedulerConfig(turn_timeout_seconds=0.05))

        async def turn(observation: Observation, use_model: bool) -> dict:
            if use_model:
                await asyncio.sleep(1)
            return {"use_model": use_model}

        [(_, result, mode)] = await scheduler.run([task_observation("slow")], turn)
        self.assertEqual("policy:timeout", mode)
        self.assertFalse(result["use_model"])

    async def test_offline_matches_never_use_the_model(self) -> None:
        plan = DecisionScheduler(model_enabled=False).plan([task_observation("a", phase=Phase.VOTING)])
        self.assertEqual("policy:offline", plan[0][1])


class ScheduledMatchTests(unittest.TestCase):
    def test_a_full_lobby_stays_within_the_model_budget(self) -> None:
        egos = list(ego_ids())
        llm = fake_llm(first_legal_answer)
        environment = SimEnvironment(egos, seed=8, impostors=2)
        record = MatchRecord("budget", 8, {e: e for e in egos}, "sim")
        scheduler = DecisionScheduler(SchedulerConfig(max_concurrent_model_turns=2, model_turns_per_tick=3))
        asyncio.run(run_match(environment, PipelineAgentRunner(llm), record, scheduler=scheduler))
        self.assertEqual("finished", record.status, record.error)
        decisions = [e for e in record.entries if e["type"] == "decision"]
        per_tick: dict[int, int] = {}
        for d in decisions:
            if d["scheduling"] == "model" and d["phase"] == "TASKS":
                per_tick[d["tick"]] = per_tick.get(d["tick"], 0) + 1
        self.assertLessEqual(max(per_tick.values()), 3)
        self.assertIn("policy:budget", record.summary()["scheduling"])
        self.assertTrue(all(d["accepted"] for d in decisions))

    def test_invalid_scheduler_settings_are_refused(self) -> None:
        with self.assertRaises(HTTPException):
            _match_options({"scheduler": {"max_concurrent_model_turns": 0}})
        with self.assertRaises(HTTPException):
            _match_options({"scheduler": {"parallel": 99}})
        self.assertEqual(4, _match_options({})["scheduler"].max_concurrent_model_turns)


class BaselineTests(unittest.TestCase):
    def _mixed(self, seed: int, llm=None) -> MatchRecord:
        egos = ["archive", "bubbles", "echo", "maverick", "sherlock", "gossip"]
        modes = {e: "baseline" if e in ("archive", "echo", "gossip") else "ego" for e in egos}
        environment = SimEnvironment(egos, seed=seed)
        record = MatchRecord(f"mix{seed}", seed, {e: e for e in egos}, "sim", modes=modes)
        asyncio.run(run_match(environment, PipelineAgentRunner(llm), record, offline=llm is None))
        return record

    def test_mixed_lobby_reports_both_memory_architectures(self) -> None:
        record = self._mixed(5)
        self.assertEqual("finished", record.status, record.error)
        report = evaluate(record)
        self.assertEqual({"ego", "baseline"}, set(report["by_memory_mode"]))
        self.assertEqual(0, report["memory_leakage"]["violations"])
        baseline = report["by_memory_mode"]["baseline"]
        self.assertEqual(["archive", "echo", "gossip"], baseline["agents"])
        # A verbatim transcript remembers every detail it saw, exactly.
        self.assertEqual(1.0, baseline["detail_accuracy_mean"])
        self.assertGreaterEqual(baseline["recall_coverage_mean"], report["by_memory_mode"]["ego"]["recall_coverage_mean"])

    def test_context_size_is_measured_per_model_turn(self) -> None:
        report = evaluate(self._mixed(6, fake_llm(first_legal_answer)))["by_memory_mode"]
        for mode in ("ego", "baseline"):
            self.assertGreater(report[mode]["model_turns"], 0)
            self.assertGreater(report[mode]["context_chars_mean"], 0)

    def test_transcript_never_forgets_and_is_bounded(self) -> None:
        from among_us.steps import TRANSCRIPT_LIMIT
        record = self._mixed(5)
        last = [e for e in record.entries if e["type"] == "decision" and e["agent_id"] == "gossip"][-1]
        self.assertEqual([], last["result"]["memory"]["forgotten"])
        self.assertLessEqual(last["result"]["memory"]["stored"], TRANSCRIPT_LIMIT)

    def test_baseline_players_must_be_in_the_match(self) -> None:
        with self.assertRaises(HTTPException):
            _match_options({"egos": ["sherlock", "archive", "bubbles", "echo"], "baseline": ["nova"]})
        options = _match_options({"egos": ["sherlock", "archive", "bubbles", "echo"], "baseline": ["echo"]})
        self.assertEqual("baseline", options["modes"]["echo"])
        self.assertEqual("ego", options["modes"]["sherlock"])


class OpportunityTests(unittest.TestCase):
    def test_a_declined_kill_opportunity_does_not_repeat_within_the_situation(self) -> None:
        engine = SimEngine(["a", "b", "c", "d"], seed=1, kill_cooldown=0)
        impostor = next(pid for pid in engine.player_ids if engine.role_of(pid) is Role.IMPOSTOR)
        for pid in engine.player_ids:
            engine._players[pid].room = "electrical"
        engine.observe(impostor)
        engine.submit(SemanticAction(impostor, ActionKind.WAIT))
        engine.advance()
        engine.observe(impostor)
        engine.submit(SemanticAction(impostor, ActionKind.MOVE_TO, "reactor"))
        self.assertFalse(engine.needs_decision(impostor), "same targets, already declined")


def claiming_answer(messages) -> str:
    """Crew tells the truth about where it stands; impostors claim medbay."""
    for message in messages:
        if isinstance(message, UntrustedContextMessage):
            for record in message.records:
                if record.record_id == "observation:self":
                    seen = json.loads(record.content)
                    legal = seen["legal_actions"]
                    you = seen["you"]
                    if not any(a["action"] == "SAY" for a in legal):
                        return "{}"   # no proposal: the policy plays the task phase
                    tick = int(json.loads(messages[-1].content)["tick"])
                    room = "medbay" if you["role"] == "IMPOSTOR" else you["room"]
                    return json.dumps({"action": "SAY", "text": f"I was in {room}.", "goal": "ALIBI",
                                       "claims": [{"player": you["player_id"], "room": room, "tick": tick}]})
    return "{}"


class ClaimTests(unittest.TestCase):
    def _validate(self, claims, evidence=()) -> tuple:
        return validate_proposal(
            {"action": "SAY", "text": "I was there.", "evidence_ids": list(evidence), "claims": claims},
            crew_observation(), set(evidence), load_ego("sherlock"), players=EGOS,
        )

    def test_a_claim_about_oneself_needs_no_evidence(self) -> None:
        action, status = self._validate([{"player": "sherlock", "room": "admin", "tick": 9}])
        self.assertEqual("accepted", status["status"])
        self.assertEqual([{"player": "sherlock", "room": "admin", "tick": 9}], status["claims"])

    def test_a_claim_about_someone_else_needs_evidence(self) -> None:
        _, status = self._validate([{"player": "echo", "room": "admin", "tick": 9}])
        self.assertEqual("unsupported", status["status"])
        _, status = self._validate([{"player": "echo", "room": "admin", "tick": 9}], evidence=["e1"])
        self.assertEqual("accepted", status["status"])

    def test_malformed_claims_are_refused(self) -> None:
        for claims in (
            [{"player": "sherlock", "room": "admin", "tick": 99}],          # future
            [{"player": "sherlock", "room": "admin", "tick": True}],        # not a tick
            [{"player": "stranger", "room": "admin", "tick": 1}],           # unknown player
            [{"player": "sherlock", "room": "Admin Room!", "tick": 1}],     # not a room name
            [{"player": "sherlock", "room": "admin", "tick": 1, "x": 1}],   # extra key
            [{"player": "sherlock", "room": "admin", "tick": 1}] * 6,       # too many
        ):
            with self.subTest(claims=claims):
                self.assertEqual("malformed", self._validate(claims)[1]["status"])

    def test_lies_detection_and_contradictions_are_scored(self) -> None:
        from among_us.environment import GroundTruth
        record = MatchRecord("claims", 1, {e: e for e in ("imp", "crew", "witness")}, "sim")
        record.ground_truth = GroundTruth(
            roles={"imp": "IMPOSTOR", "crew": "CREWMATE", "witness": "CREWMATE"},
            events=[],
            positions={t: {"imp": "electrical", "crew": "admin", "witness": "electrical"} for t in range(0, 12)},
        )

        def say(speaker, claims):
            record.entries.append({"type": "decision", "agent_id": speaker, "ego": "sherlock", "memory_mode": "ego",
                                   "tick": 10, "phase": "MEETING", "action": {"action": "SAY"}, "accepted": True,
                                   "result": {"claims": claims, "validation": {"status": "accepted"}}})

        say("imp", [{"player": "imp", "room": "medbay", "tick": 5}])        # lie, witness saw imp in electrical
        say("imp", [{"player": "imp", "room": "admin", "tick": 5}])         # second, contradicting lie
        say("crew", [{"player": "crew", "room": "admin", "tick": 6}])       # truth
        say("crew", [{"player": "imp", "room": "reactor", "tick": 6}])      # crew mistake
        report = evaluate(record)["claims"]
        self.assertEqual(2, report["impostors"]["lies"])
        self.assertEqual(1.0, report["impostors"]["detectable_lie_rate"])
        self.assertEqual(1, report["impostors"]["self_contradictions"])
        self.assertEqual(0.0, report["impostors"]["deception_consistency"])
        self.assertEqual({"claims": 2, "true": 1, "false": 1, "false_about_self": 0, "detectable_false": 1},
                         {k: report["crew"][k] for k in ("claims", "true", "false", "false_about_self", "detectable_false")})

    def test_claims_flow_from_model_to_evaluation(self) -> None:
        for seed in range(20):
            record, _ = play(seed, PipelineAgentRunner(fake_llm(claiming_answer)), offline=False)
            report = evaluate(record)["claims"]
            if report["crew"]["claims"] and report["impostors"]["claims"]:
                break
        else:
            self.fail("no seed produced claims from both sides")
        self.assertEqual(report["crew"]["claims"], report["crew"]["true"], "crew stated its true room")
        self.assertGreater(report["impostors"]["lies"], 0)


class DashboardTests(unittest.TestCase):
    def test_agent_text_is_never_inserted_as_markup(self) -> None:
        script = Path("among_us/static/app.js").read_text(encoding="utf-8")
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
            self.assertNotIn(sink, script)

    def test_dashboard_follows_the_portal_without_motion(self) -> None:
        styles = Path("among_us/static/styles.css").read_text(encoding="utf-8")
        self.assertIn("--paper: #010c0e", styles)
        for motion_rule in ("transition:", "animation:", "@keyframes", "scroll-behavior:"):
            self.assertNotIn(motion_rule, styles)

    def test_served_assets_exist_and_are_packaged(self) -> None:
        from among_us.adapter import _ASSETS

        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        patterns = project["tool"]["setuptools"]["package-data"]["among_us"]
        for name in _ASSETS:
            self.assertTrue(Path("among_us/static", name).is_file(), name)
            self.assertIn(f"static/*{Path(name).suffix}", patterns)

    def test_map_layout_covers_every_room(self) -> None:
        from among_us.sim.map import ROOMS

        script = Path("among_us/static/app.js").read_text(encoding="utf-8")
        for room in ROOMS:
            self.assertRegex(script, rf"\b{room}: \[")

    def test_spectator_view_is_not_part_of_any_observation(self) -> None:
        engine = SimEngine(EGOS, seed=1)
        view = engine.spectator_view()
        self.assertEqual(len(EGOS), len(view["players"]))
        observation = to_dict(engine.observe("sherlock"))
        self.assertNotIn("players", observation)
        self.assertNotIn("role", json.dumps(observation["visible_players"]))


class ServerContractTests(unittest.TestCase):
    """The nlght Impostor server (nlght-ai-impostor) is the bridge for the real game."""

    def _server_observation(self) -> Observation:
        return decode((FIXTURES / "server_decision_request.json").read_text(encoding="utf-8")).observation

    def test_a_decision_request_from_the_server_decodes(self) -> None:
        observation = self._server_observation()
        self.assertEqual(observation.agent_id, observation.self_state.player_id)
        perceive(observation, observation.agent_id)
        self.assertIn(observation.agent_id, observation.roster)
        self.assertTrue(observation.legal_actions)

    def test_an_agent_turn_answers_a_server_observation_with_a_legal_action(self) -> None:
        observation = self._server_observation()
        runner = PipelineAgentRunner()
        payload = {"match_id": "real", "match_seed": 1, "agent_id": observation.agent_id, "ego": "sherlock",
                   "players": [observation.agent_id], "source": "nlght-ai-impostor", "offline": True,
                   "observation": to_dict(observation)}
        result = asyncio.run(runner.decide(agent_id=observation.agent_id, session_key="s", payload=payload))
        legal = {(a.action.value, a.target) for a in observation.legal_actions}
        self.assertIn((result["action"]["action"], result["action"]["target"]), legal)
        # Humans in the roster are suspects too, not only the agents the orchestrator runs.
        self.assertIn("host", result["beliefs"])

    def test_egos_follow_the_bot_names(self) -> None:
        from among_us.match import _assign_egos

        record = MatchRecord("m", 1, {}, "bridge")
        _assign_egos(record, ["sherlock", "bubbles", "ada"])
        self.assertEqual("sherlock", record.egos["sherlock"])
        self.assertEqual("bubbles", record.egos["bubbles"])
        self.assertNotIn(record.egos["ada"], {"sherlock", "bubbles"})
        with self.assertRaises(ValueError):
            _assign_egos(MatchRecord("m", 1, {}, "bridge"), [f"bot{i}" for i in range(16)])

    def test_a_bot_named_like_an_ego_keeps_it(self) -> None:
        from among_us.match import _assign_egos

        # Found in the first end-to-end run: "dax" took the free ego "echo" before the bot named Echo.
        record = MatchRecord("m", 1, {}, "bridge")
        _assign_egos(record, ["ada", "bolt", "cleo", "dax", "echo"])
        self.assertEqual("echo", record.egos["echo"])
        self.assertEqual(5, len(set(record.egos.values())))

    def test_a_bridge_match_needs_no_player_map(self) -> None:
        options = _match_options({"environment": "bridge"})
        self.assertEqual({}, options["assignment"])


class ImpostorModeTests(unittest.TestCase):
    def _engine(self) -> tuple[SimEngine, str, list[str]]:
        engine = SimEngine(["a", "b", "c", "d", "e"], seed=2, kill_cooldown=0)
        impostor = next(p for p in engine.player_ids if engine.role_of(p) is Role.IMPOSTOR)
        crew = [p for p in engine.player_ids if p != impostor]
        return engine, impostor, crew

    def _place(self, engine: SimEngine, rooms: dict[str, str]) -> None:
        for pid, room in rooms.items():
            engine._players[pid].room = room

    def test_only_impostors_vent_and_are_seen_doing_it(self) -> None:
        engine, impostor, crew = self._engine()
        self._place(engine, {impostor: "electrical", crew[0]: "electrical", crew[1]: "medbay", crew[2]: "admin", crew[3]: "admin"})
        legal = {(a.action, a.target) for a in engine.legal_actions(impostor)}
        self.assertIn((ActionKind.VENT, "medbay"), legal)
        self.assertIn((ActionKind.VENT, "security"), legal)
        self.assertFalse(any(a.action is ActionKind.VENT for a in engine.legal_actions(crew[0])))
        self.assertTrue(engine.submit(SemanticAction(impostor, ActionKind.VENT, "medbay")).accepted)
        engine.advance()
        self.assertEqual("medbay", engine._players[impostor].room)
        seen = {pid: [e.text for e in engine.observe(pid).events if e.kind is EventKind.VENT_SEEN] for pid in crew[:2]}
        self.assertEqual({crew[0]: ["entered"], crew[1]: ["exited"]}, seen)

    def test_a_sabotage_alarms_everyone_and_the_crew_fixes_it_by_going_there(self) -> None:
        engine, impostor, crew = self._engine()
        engine._sabotage_cooldown = 0
        self.assertTrue(engine.submit(SemanticAction(impostor, ActionKind.SABOTAGE, "reactor")).accepted)
        engine.advance()
        observation = engine.observe(crew[0])
        self.assertEqual("reactor", observation.alarm)
        self.assertIn(EventKind.SABOTAGE_ALARM, [e.kind for e in observation.events])
        self.assertIn("observed", observation.decision_reason)
        # Nobody knows who pulled it.
        self.assertTrue(all(e.actor is None for e in observation.events if e.kind is EventKind.SABOTAGE_ALARM))
        self.assertEqual(ActionKind.MOVE_TO, fallback_action(observation, {}, set()).action)
        self._place(engine, {crew[0]: "reactor"})
        engine.advance()
        self.assertIsNone(engine.observe(crew[1]).alarm)
        self.assertEqual(1, sum(1 for e in engine.events if e.kind == "SABOTAGE_FIXED"))

    def test_an_unfixed_sabotage_wins_for_the_impostors(self) -> None:
        engine, impostor, crew = self._engine()
        engine._sabotage_cooldown = 0
        self._place(engine, {pid: "admin" for pid in crew})
        engine.submit(SemanticAction(impostor, ActionKind.SABOTAGE, "o2"))
        end = None
        for _ in range(15):
            end = engine.advance()
            if end:
                break
        self.assertEqual("impostors", end.winner)
        self.assertIn("o2", end.reason)

    def test_the_killer_remembers_its_kill_and_nobody_else_can(self) -> None:
        engine, impostor, crew = self._engine()
        self._place(engine, {impostor: "admin", crew[0]: "admin", crew[1]: "storage", crew[2]: "storage", crew[3]: "storage"})
        engine.submit(SemanticAction(impostor, ActionKind.KILL, crew[0]))
        engine.advance()
        observation = engine.observe(impostor)
        perceive(observation, impostor)
        own = [e for e in observation.events if e.kind is EventKind.KILL_COMMITTED]
        self.assertEqual([(impostor, crew[0])], [(e.actor, e.target) for e in own])
        self.assertIn("KILL_COMMITTED", observation.decision_reason)
        # The killer may report its own victim as cover.
        self.assertIn((ActionKind.REPORT, f"body_{crew[0]}"), {(a.action, a.target) for a in observation.legal_actions})
        leaked = crew_observation(agent_id=crew[1], self_state=SelfState(crew[1], "storage", Role.CREWMATE), events=own)
        with self.assertRaises(PerceptionViolation):
            perceive(leaked, crew[1])

    def test_seeing_someone_vent_makes_them_the_prime_suspect(self) -> None:
        vent = ObservedEvent("v1", 5, EventKind.VENT_SEEN, "electrical", "echo", text="entered")
        update = consolidate(empty_store(), [vent], load_ego("sherlock"), match_seed=1, agent_id="sherlock", now=6)
        beliefs = update_beliefs(recall(update.store, load_ego("sherlock"), 6), load_ego("sherlock"), self_id="sherlock", players=EGOS)
        self.assertEqual(max(b["suspicion"] for b in beliefs.values()), beliefs["echo"]["suspicion"])

    def test_impostor_play_is_scored_without_leakage(self) -> None:
        totals = {"kills": 0, "vent_uses": 0, "sabotages": 0}
        for seed in range(8):
            record, _ = play(seed)
            report = evaluate(record)
            self.assertEqual(0, report["memory_leakage"]["violations"], seed)
            for key in totals:
                totals[key] += report["impostor_play"][key]
        self.assertTrue(all(value > 0 for value in totals.values()), totals)


class OutageDoorAndRoleTests(unittest.TestCase):
    def _engine(self, roles=()) -> tuple[SimEngine, str, list[str]]:
        engine = SimEngine(["a", "b", "c", "d", "e"], seed=2, kill_cooldown=0, impostor_roles=roles)
        impostor = next(p for p in engine.player_ids if engine.role_of(p) is Role.IMPOSTOR)
        crew = [p for p in engine.player_ids if p != impostor]
        return engine, impostor, crew

    def _place(self, engine: SimEngine, rooms: dict[str, str]) -> None:
        for pid, room in rooms.items():
            engine._players[pid].room = room

    def _sabotage(self, engine: SimEngine, impostor: str, system: str) -> None:
        engine._sabotage_cooldown = 0
        self.assertTrue(engine.submit(SemanticAction(impostor, ActionKind.SABOTAGE, system)).accepted)
        engine.advance()

    def test_lights_out_the_crew_sees_someone_but_not_who(self) -> None:
        engine, impostor, crew = self._engine()
        self._place(engine, {impostor: "admin", crew[0]: "admin", crew[1]: "admin", crew[2]: "storage", crew[3]: "storage"})
        self._sabotage(engine, impostor, "lights")
        seen = engine.observe(crew[0])
        self.assertEqual(("lights",), seen.outages)
        self.assertIsNone(seen.alarm)
        self.assertEqual({"someone"}, {p.player_id for p in seen.visible_players})
        # Impostors are not blinded: they still recognise everyone.
        self.assertEqual({crew[0], crew[1]}, {p.player_id for p in engine.observe(impostor).visible_players})
        engine.submit(SemanticAction(impostor, ActionKind.KILL, crew[1]))
        engine.advance()
        kill = [e for e in engine.observe(crew[0]).events if e.kind is EventKind.KILL_WITNESSED]
        self.assertEqual([None], [e.actor for e in kill])
        # Not critical: no countdown, no loss; fixed in electrical.
        for _ in range(15):
            self.assertIsNone(engine.advance())
        self._place(engine, {crew[2]: "electrical"})
        engine.advance()
        self.assertEqual((), engine.observe(crew[0]).outages)

    def test_comms_down_the_crew_cannot_see_its_tasks(self) -> None:
        engine, impostor, crew = self._engine()
        self._place(engine, {pid: "admin" for pid in engine.player_ids})
        self._sabotage(engine, impostor, "comms")
        observation = engine.observe(crew[0])
        self.assertEqual((), observation.self_state.tasks_remaining)
        self.assertFalse(any(a.action is ActionKind.DO_TASK for a in observation.legal_actions))
        self.assertTrue(engine.observe(impostor).self_state.tasks_remaining)
        self._place(engine, {crew[0]: "communications"})
        engine.advance()
        self.assertTrue(engine.observe(crew[0]).self_state.tasks_remaining)

    def test_shut_doors_hold_everyone_in_and_are_heard_next_door(self) -> None:
        engine, impostor, crew = self._engine()
        self._place(engine, {impostor: "admin", crew[0]: "electrical", crew[1]: "storage", crew[2]: "cafeteria", crew[3]: "reactor"})
        self.assertTrue(engine.submit(SemanticAction(impostor, ActionKind.CLOSE_DOORS, "electrical")).accepted)
        engine.submit(SemanticAction(crew[0], ActionKind.MOVE_TO, "storage"))
        engine.advance()
        self.assertEqual("electrical", engine._players[crew[0]].room)
        heard = {pid for pid in crew if any(e.kind is EventKind.DOORS_CLOSED for e in engine.observe(pid).events)}
        self.assertEqual({crew[0], crew[1]}, heard)
        for _ in range(6):
            engine.advance()
        self.assertEqual("storage", engine._players[crew[0]].room)

    def test_a_shapeshifted_killer_is_seen_as_the_one_it_copied(self) -> None:
        engine, impostor, crew = self._engine(roles=("SHAPESHIFTER",))
        self.assertEqual("SHAPESHIFTER", engine.observe(impostor).self_state.special_role)
        self._place(engine, {impostor: "admin", crew[0]: "storage", crew[1]: "storage", crew[2]: "reactor", crew[3]: "reactor"})
        self.assertTrue(engine.submit(SemanticAction(impostor, ActionKind.SHAPESHIFT, crew[3])).accepted)
        engine.advance()
        self.assertEqual(crew[3], engine.observe(impostor).self_state.disguised_as)
        self._place(engine, {impostor: "storage"})
        self.assertEqual({crew[3], crew[1]}, {p.player_id for p in engine.observe(crew[0]).visible_players})
        engine.submit(SemanticAction(impostor, ActionKind.KILL, crew[1]))
        engine.advance()
        blamed = [e.actor for e in engine.observe(crew[0]).events if e.kind is EventKind.KILL_WITNESSED]
        self.assertEqual([crew[3]], blamed)
        # The shape wears off in front of the witness: the true face shows.
        for _ in range(9):
            engine.advance()
        reverted = [(e.actor, e.target) for e in engine.observe(crew[0]).events if e.kind is EventKind.SHAPESHIFT_SEEN]
        self.assertEqual([(crew[3], impostor)], reverted)
        record_events = [e.to_record() for e in engine.events]
        record = MatchRecord("ss", 2, {}, "sim")
        from among_us.environment import GroundTruth
        record.ground_truth = GroundTruth({p: engine.role_of(p).value for p in engine.player_ids}, record_events, engine.positions)
        self.assertEqual(1, evaluate(record)["impostor_play"]["kill_sightings_misattributed"])

    def test_a_vanished_phantom_is_not_perceived_and_cannot_kill(self) -> None:
        engine, impostor, crew = self._engine(roles=("PHANTOM",))
        self._place(engine, {impostor: "admin", crew[0]: "admin", crew[1]: "storage", crew[2]: "reactor", crew[3]: "reactor"})
        self.assertTrue(engine.submit(SemanticAction(impostor, ActionKind.VANISH)).accepted)
        engine.advance()
        observation = engine.observe(crew[0])
        self.assertIn(EventKind.VANISH_SEEN, [e.kind for e in observation.events])
        self.assertNotIn(impostor, {p.player_id for p in observation.visible_players})
        mine = engine.observe(impostor)
        self.assertTrue(mine.self_state.vanished)
        self.assertFalse(any(a.action is ActionKind.KILL for a in mine.legal_actions))
        engine.submit(SemanticAction(impostor, ActionKind.MOVE_TO, "storage"))
        engine.advance()
        self.assertEqual([], [e for e in engine.observe(crew[1]).events if e.actor == impostor])
        engine.submit(SemanticAction(impostor, ActionKind.APPEAR))
        engine.advance()
        self.assertIn(EventKind.APPEAR_SEEN, [e.kind for e in engine.observe(crew[1]).events])

    def test_a_crewmate_never_carries_impostor_abilities(self) -> None:
        with self.assertRaises(PerceptionViolation):
            perceive(crew_observation(self_state=SelfState("sherlock", "admin", Role.CREWMATE, special_role="PHANTOM")), "sherlock")

    def test_matches_with_special_roles_stay_leak_free(self) -> None:
        used = Counter()
        for seed in range(8):
            environment = SimEnvironment(EGOS, seed=seed, impostor_roles=("SHAPESHIFTER",) if seed % 2 else ("PHANTOM",))
            record = MatchRecord(f"r{seed}", seed, {e: e for e in EGOS}, "sim")
            asyncio.run(run_match(environment, PipelineAgentRunner(), record, offline=True))
            self.assertEqual("finished", record.status, record.error)
            report = evaluate(record)
            self.assertEqual(0, report["memory_leakage"]["violations"], seed)
            play = report["impostor_play"]
            used.update({key: play[key] for key in ("shapeshifts", "vanishes", "doors_closed")})
            used.update(play["sabotages_by_system"])
        self.assertTrue(all(used[key] > 0 for key in ("shapeshifts", "vanishes", "doors_closed")), used)

    def test_impostor_roles_are_validated(self) -> None:
        with self.assertRaises(HTTPException):
            _match_options({"impostor_roles": ["VIPER"]})
        self.assertEqual(["PHANTOM"], _match_options({"impostor_roles": ["PHANTOM"]})["impostor_roles"])
