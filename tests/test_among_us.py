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
from among_us.environment import SimEnvironment
from among_us.evaluation import differentiation_probe, evaluate
from among_us.match import MatchRecord, PipelineAgentRunner, run_match, session_key
from among_us.memory import consolidate, empty_store, recall
from among_us.perception import PerceptionViolation, perceive
from among_us.policy import fallback_action
from among_us.protocol import (
    ActionKind,
    ChatMessage,
    EventKind,
    LegalAction,
    LobbyRegistered,
    Observation,
    ObservedEvent,
    Phase,
    ProtocolError,
    Role,
    SeatedAgent,
    SelfState,
    SemanticAction,
    Situation,
    VisiblePlayer,
    decode,
    encode,
    registered_to_dict,
    registration_from_dict,
    to_dict,
)
from among_us.scheduler import DecisionScheduler, SchedulerConfig
from among_us.sim.engine import MAX_PLAYERS, SimEngine, max_impostors
from among_us.assess import room_claims
from among_us.turn import parse_game, parse_meeting
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


def _turn(messages) -> tuple[str, str, str]:
    """The rules, the agent's own view and the request of one model call."""
    rules = next(m.content for m in messages if isinstance(m, TrustedInstructionMessage))
    view = next(r.content for m in messages if isinstance(m, UntrustedContextMessage) for r in m.records if r.record_id == "turn:view")
    return rules, view, str(messages[-1].content)


def first_legal_answer(messages) -> str:
    """Game turn: the last option. Meeting: say something harmless, or skip the vote."""
    rules, view, request = _turn(messages)
    if "Your number" in request:
        return re.findall(r"^(\d+) ", view.split("Options:")[1], re.MULTILINE)[-1]
    if "SAY <" in rules:
        return "SAY I was doing my tasks."
    return "SKIP" if "SKIP" in rules else "WAIT"


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
    SESSION_MESSAGES = ("game_started", "observation", "decision_request", "action", "action_result",
                        "agent_left", "game_ended")

    def test_every_fixture_round_trips(self) -> None:
        for name in self.SESSION_MESSAGES:
            raw = (FIXTURES / f"{name}.json").read_text(encoding="utf-8")
            message = decode(raw)
            self.assertEqual(json.loads(raw), json.loads(encode(message)), name)

    def test_the_registration_fixtures_round_trip(self) -> None:
        raw = json.loads((FIXTURES / "lobby_registration.json").read_text(encoding="utf-8"))
        registration = registration_from_dict(raw)
        self.assertEqual(raw, {"protocol": 2, **to_dict(registration)})
        registered = json.loads((FIXTURES / "lobby_registered.json").read_text(encoding="utf-8"))
        agents = tuple(SeatedAgent(**a) for a in registered["agents"])
        self.assertEqual(registered, registered_to_dict(LobbyRegistered(registered["lobby_id"], registered["token"], agents)))
        for agent in agents:
            self.assertEqual(agent.name.lower(), agent.agent_id)

    def test_an_observation_that_discloses_a_role_is_refused(self) -> None:
        raw = (FIXTURES / "invalid_leaky_observation.json").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "unknown fields"):
            decode(raw)

    def test_version_and_type_are_checked(self) -> None:
        with self.assertRaises(ProtocolError):
            decode(json.dumps({"type": "game_started", "protocol": 1, "map": "skeld"}))
        with self.assertRaises(ProtocolError):
            decode(json.dumps({"type": "teleport", "protocol": 2}))
        with self.assertRaises(ProtocolError):
            registration_from_dict({"protocol": 2, "game_code": "X", "game_version": "v", "source": "s", "count": "5"})

    def test_a_decision_request_needs_an_id_and_a_deadline(self) -> None:
        raw = json.loads((FIXTURES / "decision_request.json").read_text(encoding="utf-8"))
        for broken in ({**raw, "deadline_seconds": 0}, {k: v for k, v in raw.items() if k != "request_id"}):
            with self.assertRaises(ProtocolError):
                decode(json.dumps(broken))

    def test_an_action_on_the_agents_own_initiative_has_no_request(self) -> None:
        message = decode(json.dumps({"type": "action", "protocol": 2, "request_id": None, "agent_id": "sherlock",
                                     "action": "CALL_MEETING", "target": None, "text": None, "goal": "ACCUSE",
                                     "evidence_ids": []}))
        self.assertIsNone(message.request_id)
        self.assertEqual(ActionKind.CALL_MEETING, message.action.action)


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


class AnswerTests(unittest.TestCase):
    """A game turn answers with a number, a meeting turn with a line; nothing else needs parsing."""

    def test_a_game_answer_is_an_option_number(self) -> None:
        options = [LegalAction(ActionKind.MOVE_TO, "admin"), LegalAction(ActionKind.WAIT)]
        self.assertEqual((ActionKind.WAIT, None), parse_game("2", options, "sherlock").action.legal_key())
        self.assertEqual((ActionKind.MOVE_TO, "admin"), parse_game("Option 1.", options, "sherlock").action.legal_key())
        self.assertEqual("illegal", parse_game("7", options, "sherlock").status)
        self.assertEqual("no_proposal", parse_game("I would rather not.", options, "sherlock").status)

    def test_a_meeting_answer_is_its_last_line(self) -> None:
        refs = {"m1": "e1", "m2": "e2"}
        answer = parse_meeting("Hawk said medbay, but I saw him elsewhere.\n**SAY:** Hawk, you were in electrical. | m2 m9",
                               crew_observation(), refs, load_ego("sherlock"))
        self.assertEqual("accepted", answer.status)
        self.assertEqual("Hawk, you were in electrical.", answer.action.text)
        self.assertEqual(("e2",), answer.action.evidence_ids, "an unknown memory number is dropped")

    def test_the_view_of_others_may_change(self) -> None:
        observation = crew_observation(roster=("sherlock", "hawk", "echo"))
        answer = parse_meeting("WAIT\nMIND hawk+ echo= nobody-", observation, {}, load_ego("sherlock"))
        self.assertIs(ActionKind.WAIT, answer.action.action)
        self.assertEqual((("hawk", "+"), ("echo", "trust")), answer.mind)

    def test_what_the_phase_does_not_allow_is_refused(self) -> None:
        self.assertEqual("illegal", parse_meeting("VOTE hawk", crew_observation(), {}, load_ego("sherlock")).status)
        self.assertEqual("no_proposal", parse_meeting("Hmm.", crew_observation(), {}, load_ego("sherlock")).status)

    def test_memory_numbers_and_ticks_never_reach_the_chat(self) -> None:
        refs = {"m5": "e5", "m6": "e6"}
        answer = parse_meeting("SAY Hawk, I saw you in electrical at t20 (m5, m6). | m5", crew_observation(), refs, load_ego("sherlock"))
        self.assertEqual("Hawk, I saw you in electrical.", answer.action.text)
        self.assertEqual(("e5",), answer.action.evidence_ids)
        self.assertEqual("no_proposal", parse_meeting("SAY m5 m6", crew_observation(), refs, load_ego("sherlock")).status)

    def test_memories_are_told_in_plain_time(self) -> None:
        from among_us.turn import describe

        seen = {"tick": 18, "kind": "PLAYER_LEFT", "details": {"actor": "hawk", "room": "electrical"}, "certainty": 0.9}
        self.assertEqual("hawk left in electrical, just now (sure)", describe(seen, 20))
        self.assertNotRegex(describe(seen, 60), r"\bt\d+")

    def test_long_statements_are_cut_to_the_ego(self) -> None:
        answer = parse_meeting("SAY " + "word " * 100, crew_observation(), {}, load_ego("bubbles"))
        self.assertLessEqual(len(answer.action.text.split()), load_ego("bubbles").voice.max_words + 1)


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
        origins = {
            item.record_id: {ref.kind for ref in item.provenance.origin.sources}
            for messages in calls for m in messages if isinstance(m, UntrustedContextMessage) for item in m.records
        }
        self.assertEqual({"game_observation"}, origins["turn:view"])
        for record_id in ("turn:chat", "turn:heard"):
            if record_id in origins:
                self.assertEqual({"game_chat"}, origins[record_id], "other players' words stay marked as theirs")
        self.assertEqual("finished", record.status, record.error)

    def test_garbage_from_the_model_falls_back_to_a_legal_action(self) -> None:
        record, _ = play(2, PipelineAgentRunner(fake_llm(lambda _: '{"action":"TELEPORT"}')), offline=False)
        self.assertEqual("finished", record.status, record.error)
        decisions = [e for e in record.entries if e["type"] == "decision"]
        self.assertTrue(all(d["accepted"] for d in decisions))
        # Not an option number: not understood, so the agent's plan or the policy decided instead.
        self.assertTrue(any(d["result"]["validation"]["status"] == "no_proposal" for d in decisions))
        self.assertTrue({d["result"]["decided_by"] for d in decisions} <= {"plan", "policy", "model"})

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


def _sql_only(sql: str) -> str:
    return "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))


class WorkflowSeedTests(unittest.TestCase):
    def _seed(self) -> str:
        return Path("seed/08_among_us.sql").read_text(encoding="utf-8")

    def test_every_seeded_step_type_is_registered(self) -> None:
        import main  # noqa: F401
        from nlght.adapters.outbound.workflow.registry import step_registry

        seeded = set(re.findall(r"'(among_us_[a-z]+)',\n", self._seed()))
        self.assertEqual(6, len(seeded))
        self.assertEqual(set(), seeded - set(getattr(step_registry, "_registry")))

    def test_no_step_pins_a_model_provider(self) -> None:
        # Every turn names its agent's provider and model; a pinned one would win over it.
        seed = self._seed()
        self.assertNotIn("model_provider", _sql_only(seed))
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
    def test_the_real_game_is_not_started_as_a_match(self) -> None:
        with self.assertRaisesRegex(HTTPException, "") as caught:
            _match_options({"environment": "bridge"})
        self.assertIn("!bots", caught.exception.detail)

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
    """Crew tells the truth about where it stood; impostors claim medbay. Game turns are left to the plan."""
    rules, view, request = _turn(messages)
    if "SAY <" not in rules:
        return "SKIP" if "Your vote" in request else ""
    you = re.search(r"You: \S+, (\w+).*?you were in ([a-z0-9 ]+) when", view)
    room = "medbay" if you.group(1) == "impostor" else you.group(2)
    return f"SAY I was in {room}."


class ClaimTests(unittest.TestCase):
    def test_a_statement_says_where_whom_it_names_was(self) -> None:
        rooms = ("admin", "electrical", "lower_engine")
        self.assertEqual([("sherlock", "admin")], room_claims("I was in admin.", "sherlock", EGOS, rooms))
        self.assertEqual([("echo", "electrical")], room_claims("I saw echo in electrical!", "sherlock", EGOS, rooms))
        self.assertEqual([("sherlock", "lower_engine")], room_claims("Lower engine, doing wires.", "sherlock", EGOS, rooms))
        self.assertEqual([], room_claims("Where were you?", "sherlock", EGOS, rooms))

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


class MeetingConversationTests(unittest.TestCase):
    """A meeting is a conversation: agents see what was said, answer, take sides, ask — and never repeat themselves."""

    def _meeting(self, *chat: tuple[str, str], role: Role = Role.CREWMATE, partners: tuple[str, ...] = ()) -> Observation:
        return crew_observation(
            self_state=SelfState("sherlock", "admin", role, known_impostors=partners),
            visible_players=tuple(VisiblePlayer(p, "meeting") for p in ("hawk", "nova", "echo")),
            meeting_chat=tuple(ChatMessage(speaker, text) for speaker, text in chat),
        )

    def test_the_meeting_chat_travels_strictly(self) -> None:
        raw = json.loads((FIXTURES / "decision_request.json").read_text(encoding="utf-8"))
        raw["observation"]["meeting_chat"] = [{"speaker": "hawk", "text": "Sherlock, where were you?"}]
        self.assertEqual("hawk", decode(json.dumps(raw)).observation.meeting_chat[0].speaker)
        raw["observation"]["meeting_chat"] = [{"speaker": "hawk", "text": "hi", "role": "IMPOSTOR"}]
        with self.assertRaisesRegex(ProtocolError, "unknown fields"):
            decode(json.dumps(raw))

    def test_an_accused_agent_defends_itself_by_name(self) -> None:
        action = fallback_action(self._meeting(("hawk", "I think Sherlock is sus, he was alone.")), {}, set())
        self.assertEqual(ActionKind.SAY, action.action)
        self.assertTrue(action.text.startswith("hawk, it wasn't me"), action.text)

    def test_an_agent_takes_a_side_on_someone_elses_accusation(self) -> None:
        chat = ("hawk", "I saw nova near the body, nova is sus.")
        agree = fallback_action(self._meeting(chat), {"nova": {"suspicion": 0.5, "evidence_ids": []}}, set())
        self.assertEqual("I agree with hawk about nova.", agree.text)
        doubt = fallback_action(self._meeting(chat), {"nova": {"suspicion": 0.2, "evidence_ids": []}}, set())
        self.assertEqual("hawk, what makes you sure about nova?", doubt.text)

    def test_an_impostor_never_piles_onto_its_partner(self) -> None:
        action = fallback_action(self._meeting(("hawk", "nova is sus."), role=Role.IMPOSTOR, partners=("nova",)), {}, set())
        self.assertNotIn("agree", (action.text or "").lower())

    def test_no_line_twice_then_silence(self) -> None:
        said: list[tuple[str, str]] = []
        for _ in range(5):
            action = fallback_action(self._meeting(*said), {}, set())
            if action.action is ActionKind.WAIT:
                break
            said.append(("sherlock", action.text))
        self.assertEqual(len(said), len({text for _, text in said}))
        self.assertIs(ActionKind.WAIT, action.action)

    def test_the_model_sees_the_conversation_and_its_own_lines(self) -> None:
        llm = fake_llm(first_legal_answer)
        observation = self._meeting(("sherlock", "In admin."), ("hawk", "Sherlock, in admin doing what?"))
        payload = {"match_id": "m", "match_seed": 1, "agent_id": "sherlock", "ego": "sherlock", "players": ["hawk", "nova", "echo"],
                   "source": "test", "observation": to_dict(observation)}
        asyncio.run(PipelineAgentRunner(llm).decide(agent_id="sherlock", session_key="s", payload=payload))
        [messages] = llm._backend.calls
        trusted = next(m for m in messages if isinstance(m, TrustedInstructionMessage))
        self.assertIn("This is a meeting", trusted.content)
        chat = next(r for m in messages if isinstance(m, UntrustedContextMessage) for r in m.records if r.record_id == "turn:chat")
        self.assertIn("you: In admin.\nhawk: Sherlock, in admin doing what?", chat.content)
        self.assertIn('Said to you: hawk: "Sherlock, in admin doing what?"', str(messages[-1].content))
        self.assertEqual({"game_chat"}, {ref.kind for ref in chat.provenance.origin.sources}, "other players' words stay marked")
        self.assertNotIn("{", chat.content, "plain text, no JSON")



class AssessmentTests(unittest.TestCase):
    """The weighing happens before the model: situation, stance, contradictions, tactics and a held plan."""

    def _task_observation(self, *, role: Role = Role.CREWMATE, tasks: tuple[str, ...] = (), see: tuple[str, ...] = ("echo",),
                          legal: tuple[LegalAction, ...] = (), roster: tuple[str, ...] = ("sherlock", "echo", "hawk", "nova", "bubbles"),
                          kill_ready: bool = False, events: tuple[ObservedEvent, ...] = ()) -> Observation:
        return Observation(
            agent_id="sherlock", tick=20, phase=Phase.TASKS,
            self_state=SelfState("sherlock", "admin", role, tasks_remaining=tasks, kill_ready=kill_ready),
            visible_players=tuple(VisiblePlayer(p, "admin") for p in see), events=events,
            legal_actions=legal + tuple(LegalAction(ActionKind.FOLLOW, p) for p in see) + (LegalAction(ActionKind.GUARD), LegalAction(ActionKind.WAIT)),
            roster=roster,
        )

    def test_with_tasks_done_a_bold_agent_shadows_its_suspect(self) -> None:
        from among_us.assess import assess

        beliefs = {"echo": {"suspicion": 0.7, "trust": 0.0, "evidence_ids": ["e1"]}}
        result = assess(self._task_observation(), beliefs, [], load_ego("sherlock"), None)
        self.assertEqual(LegalAction(ActionKind.FOLLOW, "echo"), result.preferred)
        self.assertEqual("shadow", result.plan["tactic"])
        timid = assess(self._task_observation(), beliefs, [], load_ego("echo"), None).plan
        self.assertEqual("watch", timid["tactic"], "a timid, sociable ego stays with the others instead")

    def test_late_in_the_game_an_agent_sticks_with_someone_it_trusts(self) -> None:
        from among_us.assess import assess

        beliefs = {"echo": {"suspicion": 0.2, "trust": 0.9, "evidence_ids": []}}
        observation = self._task_observation(tasks=("wires",), roster=("sherlock", "echo", "hawk"))
        result = assess(observation, beliefs, [], load_ego("guardian"), None)
        self.assertEqual(("escort", LegalAction(ActionKind.FOLLOW, "echo")), (result.plan["tactic"], result.preferred))

    def test_a_plan_holds_until_its_end_condition(self) -> None:
        from among_us.assess import assess

        plan = {"tactic": "escort", "action": "FOLLOW", "target": "echo", "reason": "r", "ends_if": "e", "since": 3}
        trusted = {"echo": {"suspicion": 0.2, "trust": 0.9, "evidence_ids": []}}
        kept = assess(self._task_observation(), trusted, [], load_ego("sherlock"), plan)
        self.assertEqual((3, LegalAction(ActionKind.FOLLOW, "echo")), (kept.plan["since"], kept.preferred))
        turned = {"echo": {"suspicion": 0.8, "trust": 0.0, "evidence_ids": ["e1"]}}
        dropped = assess(self._task_observation(), turned, [], load_ego("sherlock"), plan)
        self.assertNotEqual("escort", (dropped.plan or {}).get("tactic"))

    def test_an_impostor_stalks_a_lone_crewmate_and_strikes_when_alone(self) -> None:
        from among_us.assess import assess

        two = self._task_observation(role=Role.IMPOSTOR, see=("echo", "hawk"), kill_ready=True,
                                     legal=(LegalAction(ActionKind.KILL, "echo"), LegalAction(ActionKind.KILL, "hawk")))
        self.assertEqual("stalk", assess(two, {}, [], load_ego("maverick"), None).plan["tactic"])
        one = self._task_observation(role=Role.IMPOSTOR, see=("echo",), kill_ready=True, legal=(LegalAction(ActionKind.KILL, "echo"),))
        self.assertEqual(LegalAction(ActionKind.KILL, "echo"), assess(one, {}, [], load_ego("maverick"), None).preferred)

    def test_a_lie_in_the_meeting_becomes_a_conclusion_kept_as_memory(self) -> None:
        from among_us.assess import assess, conclusions
        from among_us.beliefs import update_beliefs
        from among_us.memory import keep

        seen = {"event_id": "e5", "tick": 18, "kind": "PLAYER_LEFT", "details": {"actor": "hawk", "room": "electrical"}, "certainty": 0.9}
        meeting = crew_observation(tick=20, meeting_chat=(ChatMessage("hawk", "I was in medbay the whole time."),),
                                   roster=("sherlock", "hawk", "echo"))
        result = assess(meeting, {}, [seen], load_ego("sherlock"), None)
        [found] = result.contradictions
        self.assertEqual(("hawk", "medbay", "electrical"), (found.speaker, found.said, found.seen))
        store = keep({"memories": {}}, conclusions("sherlock", result.contradictions))
        [judged] = store["memories"].values()
        before = update_beliefs([seen], load_ego("sherlock"), self_id="sherlock", players=["hawk", "echo"])
        after = update_beliefs([seen, {**judged, "certainty": 1.0}], load_ego("sherlock"), self_id="sherlock", players=["hawk", "echo"])
        self.assertGreater(after["hawk"]["suspicion"], before["hawk"]["suspicion"])

    def test_a_game_turn_is_short_text_with_numbered_options_plan_first(self) -> None:
        llm = fake_llm(first_legal_answer)
        beliefs_echo = ObservedEvent("sherlock-00001", 1, EventKind.TASK_SEEN, room="admin", actor="echo")
        observation = self._task_observation(events=(beliefs_echo,), roster=("sherlock", "echo", "hawk"))
        payload = {"match_id": "m", "match_seed": 1, "agent_id": "sherlock", "ego": "guardian",
                   "players": ["sherlock", "echo", "hawk"], "source": "test", "observation": to_dict(observation)}
        result = asyncio.run(PipelineAgentRunner(llm).decide(agent_id="sherlock", session_key="s", payload=payload))
        rules, view, request = _turn(llm._backend.calls[0])
        self.assertIn("Answer with the number", rules)
        self.assertNotIn("{", view)
        self.assertRegex(view, r"Options:\n1 ")
        self.assertEqual({"perceive", "remember", "believe", "decide", "model", "validate"}, set(result["timings"]))


class AgendaTests(unittest.TestCase):
    """What an agent experiences becomes points to tell and to ask; at the table it answers, tells, asks, weighs."""

    seen = {"event_id": "e1", "tick": 10, "kind": "PLAYER_LEFT", "details": {"actor": "hawk", "room": "electrical"}, "certainty": 0.9}
    body = {"event_id": "e2", "tick": 14, "kind": "BODY_FOUND", "details": {"target": "nova", "room": "electrical"}, "certainty": 1.0}

    def _meeting(self, *chat: tuple[str, str]) -> Observation:
        return crew_observation(tick=15, roster=("sherlock", "hawk", "nova", "echo"),
                                meeting_chat=tuple(ChatMessage(s, t) for s, t in chat))

    def test_a_round_leaves_points_to_tell_and_to_ask(self) -> None:
        from among_us.agenda import update

        items = update([], self._meeting(), [self.seen, self.body], {})
        self.assertEqual(
            [("tell", "I saw hawk in electrical just before nova's body was found there."), ("ask", "hawk, what were you doing in electrical?")],
            [(i["kind"], i["text"]) for i in items])
        self.assertEqual([], update(items, self._meeting(), [], {}), "forgetting the memories drops the points")

    def test_what_was_said_is_settled_and_answers_are_weighed(self) -> None:
        from among_us.agenda import replies, update

        items = update([], self._meeting(), [self.seen, self.body], {})
        chat = self._meeting(("sherlock", "hawk, what were you doing in electrical?"), ("hawk", "I was in admin the whole time."))
        items = update(items, chat, [self.seen, self.body], {})
        self.assertEqual({"ask": True, "tell": False}, {i["kind"]: i["done"] for i in items}, "only what was said is settled")
        [reply] = replies(items, chat, [self.seen, self.body])
        self.assertEqual(("hawk", "contradicts", "electrical"), (reply.player, reply.verdict, reply.seen))

    def test_a_question_put_to_the_agent_is_answered_first(self) -> None:
        from among_us.agenda import questions_for, update

        recalled = [self.seen, self.body]
        asked = self._meeting(("hawk", "Sherlock, where were you?"))
        self.assertEqual(["hawk"], [q.asker for q in questions_for(asked)])
        agenda = update([], asked, recalled, {})
        self.assertEqual("hawk, I was in cafeteria.", fallback_action(asked, {}, {"e1", "e2"}, agenda).text)

        # Answering hawk does not settle the point about hawk: that comes next.
        answered = self._meeting(("hawk", "Sherlock, where were you?"), ("sherlock", "hawk, I was in cafeteria."))
        self.assertEqual([], questions_for(answered))
        agenda = update(agenda, answered, recalled, {})
        self.assertEqual("I saw hawk in electrical just before nova's body was found there.",
                         fallback_action(answered, {}, {"e1", "e2"}, agenda).text)

    def test_the_meeting_turn_shows_the_agenda_and_the_questions(self) -> None:
        from among_us.agenda import update
        from among_us.assess import assess
        from among_us.turn import meeting_prompt

        observation = self._meeting(("hawk", "Sherlock, where were you?"))
        agenda = update([], observation, [self.seen, self.body], {})
        prompt = meeting_prompt(observation, load_ego("sherlock"), assess(observation, {}, [self.seen, self.body], load_ego("sherlock"), None),
                                [self.seen, self.body], {}, agenda)
        text = "\n".join(t for _, t, _ in prompt.context)
        self.assertIn("Tell: I saw hawk in electrical just before nova's body was found there.", text)
        self.assertIn('Said to you: hawk: "Sherlock, where were you?"', prompt.request)
        self.assertIn("Answer hawk directly first", prompt.request)
        self.assertIn("First answer what you were asked", prompt.rules)


class MeetingSenseTests(unittest.TestCase):
    """Times matter, the dead are out, votes need reasons, and the meeting decides what comes after it."""

    def test_an_earlier_sighting_does_not_make_a_liar(self) -> None:
        from among_us.assess import contradictions

        body = {"event_id": "b", "tick": 30, "kind": "BODY_FOUND", "details": {"target": "nova", "room": "admin"}, "certainty": 1.0}
        earlier = {"event_id": "e", "tick": 12, "kind": "PLAYER_LEFT", "details": {"actor": "hawk", "room": "electrical"}, "certainty": 1.0}
        then = {"event_id": "n", "tick": 28, "kind": "PLAYER_LEFT", "details": {"actor": "hawk", "room": "electrical"}, "certainty": 1.0}
        meeting = crew_observation(tick=31, roster=("sherlock", "hawk", "echo"),
                                   meeting_chat=(ChatMessage("hawk", "I was in medbay."),))
        self.assertEqual([], contradictions(meeting, [body, earlier]), "hours apart, both can be true")
        self.assertEqual(["n"], [c.memory_id for c in contradictions(meeting, [body, then])])

    def test_whoever_is_not_at_the_table_is_dead_and_off_the_agenda(self) -> None:
        from among_us.agenda import update
        from among_us.assess import assess

        meeting = crew_observation(roster=("sherlock", "hawk", "nova", "echo"),
                                   visible_players=(VisiblePlayer("hawk", "meeting"), VisiblePlayer("echo", "meeting")))
        result = assess(meeting, {}, [], load_ego("sherlock"), None)
        self.assertEqual((["echo", "hawk", "sherlock"], ["nova"]), (result.alive, result.dead))
        item = {"id": "ask:nova", "kind": "ask", "about": "nova", "text": "nova, where were you?", "evidence": [], "rank": 3, "done": False}
        self.assertEqual([], update([item], meeting, [], {}, dead=result.dead))

    def test_the_meeting_decides_what_comes_after_it(self) -> None:
        meeting = crew_observation(roster=("sherlock", "echo"), visible_players=(VisiblePlayer("echo", "meeting"),))
        answer = parse_meeting("SAY Echo, let's stick together.\nMIND echo=\nNEXT FOLLOW Echo", meeting, {}, load_ego("sherlock"))
        self.assertEqual(LegalAction(ActionKind.FOLLOW, "echo"), answer.then)
        self.assertEqual((("echo", "trust"),), answer.mind)
        self.assertTrue(parse_meeting("WAIT\nNEXT TASKS", meeting, {}, load_ego("sherlock")).back_to_tasks)
        self.assertIsNone(parse_meeting("WAIT\nNEXT FOLLOW nova", meeting, {}, load_ego("sherlock")).then, "only someone at the table")

    def test_then_travels_with_the_answer(self) -> None:
        from among_us.protocol import ActionMessage

        message = ActionMessage("r1", SemanticAction("sherlock", ActionKind.SAY, text="Let's stick together."), LegalAction(ActionKind.FOLLOW, "echo"))
        self.assertEqual(message, decode(encode(message)))
        self.assertEqual({"action": "FOLLOW", "target": "echo"}, json.loads(encode(message))["then"])



class EmergencyMeetingTests(unittest.TestCase):
    """An emergency meeting needs something new seen since the last one, not the talk of the last meeting."""

    def _idle_crewmate(self, since_meeting: int) -> Observation:
        return Observation(
            agent_id="sherlock", tick=40, phase=Phase.TASKS, self_state=SelfState("sherlock", "cafeteria", Role.CREWMATE),
            legal_actions=(LegalAction(ActionKind.CALL_MEETING), LegalAction(ActionKind.WAIT)), roster=("sherlock", "hawk", "echo"),
            situation=Situation(emergencies_left=1, since_meeting=since_meeting),
        )

    def _beliefs(self, *ids: str) -> dict[str, dict[str, Any]]:
        return {"hawk": {"suspicion": 0.9, "trust": 0.0, "evidence_ids": list(ids)}}

    def test_talk_and_conclusions_from_the_last_meeting_call_no_new_one(self) -> None:
        from among_us.assess import assess
        from among_us.initiative import initiative_action

        said = {"event_id": "c1", "tick": 30, "kind": "CHAT", "details": {"actor": "echo", "text": "hawk is sus"}, "certainty": 1.0}
        judged = {"event_id": "sherlock:judged:x", "tick": 30, "kind": "JUDGED", "details": {"target": "hawk", "text": "+"}, "certainty": 1.0}
        observation = self._idle_crewmate(since_meeting=5)
        self.assertNotEqual("accuse", (assess(observation, self._beliefs("c1", "sherlock:judged:x"), [said, judged], load_ego("maverick"), None).plan or {}).get("tactic"))
        self.assertIsNone(initiative_action(observation, self._beliefs("c1", "sherlock:judged:x"), [said, judged], load_ego("maverick"), {})[0])

    def test_a_vent_seen_since_the_last_meeting_does(self) -> None:
        from among_us.assess import assess

        vent = {"event_id": "v1", "tick": 38, "kind": "VENT_SEEN", "details": {"actor": "hawk", "room": "admin"}, "certainty": 1.0}
        self.assertEqual("accuse", assess(self._idle_crewmate(since_meeting=5), self._beliefs("v1"), [vent], load_ego("maverick"), None).plan["tactic"])
        old = {**vent, "tick": 20}
        self.assertNotEqual("accuse", (assess(self._idle_crewmate(since_meeting=5), self._beliefs("v1"), [old], load_ego("maverick"), None).plan or {}).get("tactic"),
                            "seen before the last meeting: that meeting was the place for it")


class ListeningTests(unittest.TestCase):
    """Who has nothing to add listens without a model call; the meeting's cause is always in view."""

    def _table(self, *chat: tuple[str, str]) -> Observation:
        return crew_observation(roster=("sherlock", "hawk", "nova", "echo"),
                                visible_players=tuple(VisiblePlayer(p, "meeting") for p in ("hawk", "echo")),
                                meeting_chat=tuple(ChatMessage(s, t) for s, t in chat),
                                situation=Situation(called_by="hawk", body_of="nova"))

    def test_an_agent_with_nothing_to_add_listens(self) -> None:
        from among_us.assess import assess
        from among_us.turn import worth_a_turn

        def worth(observation: Observation, beliefs=None) -> bool:
            beliefs = beliefs or {}
            return worth_a_turn(observation, [], assess(observation, beliefs, [], load_ego("sherlock"), None), beliefs)

        self.assertTrue(worth(self._table()), "the opening line")
        said = self._table(("sherlock", "I was in cafeteria."), ("echo", "I was in admin."))
        self.assertFalse(worth(said))
        self.assertTrue(worth(self._table(("sherlock", "I was in cafeteria."), ("echo", "Sherlock is lying."))), "spoken to")

    def test_the_talk_turning_to_a_suspect_brings_the_agent_in(self) -> None:
        from among_us.assess import assess
        from among_us.turn import worth_a_turn

        observation = self._table(("sherlock", "I was in cafeteria."), ("echo", "hawk was with me in admin."))
        beliefs = {"hawk": {"suspicion": 0.7, "trust": 0.0, "evidence_ids": []}}
        self.assertTrue(worth_a_turn(observation, [], assess(observation, beliefs, [], load_ego("sherlock"), None), beliefs))

    def test_the_meeting_cause_heads_the_meeting_turn(self) -> None:
        from among_us.assess import assess
        from among_us.turn import meeting_prompt

        observation = self._table()
        prompt = meeting_prompt(observation, load_ego("sherlock"), assess(observation, {}, [], load_ego("sherlock"), None), [], {})
        self.assertTrue(prompt.context[0][1].startswith("This meeting: hawk reported the dead body of nova."))
