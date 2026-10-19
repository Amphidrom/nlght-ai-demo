from __future__ import annotations

import asyncio
import json
import random
import socket
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import uvicorn
from fastapi import FastAPI

from among_us.adapter import AmongUsAdapter
from among_us.adapter import _seed_problem
from among_us.ego import InvalidEgo, load_ego, parse_ego
from among_us.initiative import initiative_action
from among_us.lobby import LobbyError, LobbyRegistry
from among_us.match import PipelineAgentRunner, session_key
from among_us.models import ModelCatalog, ModelChoice, ModelChoiceError
from among_us.protocol import (
    ActionKind,
    ActionResult,
    AgentLeft,
    DecisionRequest,
    GameEnded,
    GameStarted,
    LegalAction,
    LobbyRegistration,
    Observation,
    ObservationUpdate,
    ObservedEvent,
    EventKind,
    Phase,
    Role,
    SelfState,
    SemanticAction,
    decode,
    encode,
    to_dict,
)
from among_us.steps import MEMORY_SLOT

FIXTURES = Path("among_us/protocol_fixtures")


def registration(count: int = 2, players: tuple[str, ...] = ("Host",)) -> LobbyRegistration:
    return LobbyRegistration("REDSUS", "2026.7.15", "nlght-ai-impostor", count, players)


def observation(agent_id: str, tick: int = 2, *, events: tuple[ObservedEvent, ...] = (),
                legal: tuple[LegalAction, ...] = (LegalAction(ActionKind.MOVE_TO, "admin"), LegalAction(ActionKind.WAIT)),
                reason: str = "goal reached") -> Observation:
    return Observation(
        agent_id=agent_id, tick=tick, phase=Phase.TASKS,
        self_state=SelfState(agent_id, "cafeteria", Role.CREWMATE, tasks_remaining=("fix_wiring#0",)),
        events=events, legal_actions=legal, decision_reason=reason, roster=(agent_id, "host"),
    )


def body_found(agent_id: str, n: int = 1) -> ObservedEvent:
    return ObservedEvent(f"{agent_id}-{n:05d}", 1, EventKind.BODY_FOUND, room="admin", target="host")


class SlowModelRunner(PipelineAgentRunner):
    """A model turn takes ``delay`` seconds; a policy turn is immediate."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay
        self.started = 0

    async def decide(self, *, agent_id: str, session_key: str, payload: dict[str, Any], mode: str = "ego") -> dict[str, Any]:
        if not payload.get("offline"):
            self.started += 1
            await asyncio.sleep(self.delay)
        return await super().decide(agent_id=agent_id, session_key=session_key, payload=payload, mode=mode)


def outbox(lobby) -> list[Any]:
    messages = []
    while not lobby.outbox.empty():
        messages.append(decode(lobby.outbox.get_nowait()))
    return messages


class RegistrationTests(unittest.TestCase):
    def test_agents_are_named_by_their_ego_and_never_like_a_player(self) -> None:
        registry = LobbyRegistry(PipelineAgentRunner(), offline=True, rng=random.Random(1))
        registered = registry.register(registration(15 - 1, players=("Sherlock",)))
        ids = [a.agent_id for a in registered.agents]
        self.assertEqual(14, len(set(ids)))
        self.assertNotIn("sherlock", ids, "a human called Sherlock keeps the name")
        for agent in registered.agents:
            self.assertEqual(agent.name.lower(), agent.agent_id)
            self.assertEqual(agent.ego, agent.agent_id)

    def test_more_bots_join_without_reusing_an_ego(self) -> None:
        registry = LobbyRegistry(PipelineAgentRunner(), offline=True)
        registered = registry.register(registration(3))
        lobby = registry.lobby(registered.lobby_id)
        more = registry.add_agents(lobby, 4, ("Host",))
        self.assertEqual(7, len({a.agent_id for a in (*registered.agents, *more)}))
        with self.assertRaises(LobbyError):
            registry.add_agents(lobby, 9, ("Host",))

    def test_every_lobby_has_its_own_token(self) -> None:
        registry = LobbyRegistry(PipelineAgentRunner(), offline=True)
        first, second = registry.register(registration()), registry.register(registration())
        self.assertNotEqual(first.token, second.token)
        registry.lobby(first.lobby_id, first.token)
        with self.assertRaises(LobbyError) as caught:
            registry.lobby(first.lobby_id, second.token)
        self.assertEqual(403, caught.exception.status)

    def test_two_lobbies_may_seat_the_same_ego(self) -> None:
        registry = LobbyRegistry(PipelineAgentRunner(), offline=True)
        first = {a.ego for a in registry.register(registration(15, players=())).agents}
        second = {a.ego for a in registry.register(registration(15, players=())).agents}
        self.assertEqual(first, second)


class LobbySessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.runner = PipelineAgentRunner()
        self.started: list[str] = []
        self.finished: list[str] = []
        self.registry = LobbyRegistry(
            self.runner, offline=True,
            on_match_started=lambda r: self.started.append(r.match_id),
            on_match_finished=lambda r: self.finished.append(r.match_id),
        )
        registered = self.registry.register(registration(2))
        self.lobby = self.registry.lobby(registered.lobby_id)
        self.a, self.b = (agent.agent_id for agent in registered.agents)

    async def test_a_game_is_a_match_and_observations_write_memory_without_a_decision(self) -> None:
        self.registry.handle(self.lobby, GameStarted("skeld", (self.a, self.b, "host")))
        match = self.lobby.match
        self.assertEqual("in_game", self.lobby.state)
        self.assertEqual([match.match_id], self.started)

        self.registry.handle(self.lobby, ObservationUpdate(observation(self.a, events=(body_found(self.a),), legal=())))
        await self.registry.drain()
        memory = self.runner.sessions[session_key(match.match_id, self.a)].slots[MEMORY_SLOT]
        self.assertIn(f"{self.a}-00001", memory["memories"])
        self.assertEqual([], outbox(self.lobby), "an observation is not answered")

        self.registry.handle(self.lobby, DecisionRequest("r1", 30, observation(self.a)))
        await self.registry.drain()
        [answer] = outbox(self.lobby)
        self.assertEqual("r1", answer.request_id)
        self.assertEqual(self.a, answer.action.agent_id)
        self.assertIn(answer.action.legal_key(), {(ActionKind.MOVE_TO, "admin"), (ActionKind.WAIT, None)})

        self.registry.handle(self.lobby, ActionResult(self.a, False, "no decision pending", "r1"))
        [decision] = [e for e in match.entries if e["type"] == "decision"]
        self.assertEqual((False, "no decision pending"), (decision["accepted"], decision["rejection"]))
        self.assertEqual(1, self.lobby.agents[self.a].refused)
        self.assertEqual("CREWMATE", self.lobby.agents[self.a].role)

        self.registry.handle(self.lobby, GameEnded("crew", "CrewmatesByVote", {self.a: "CREWMATE", self.b: "IMPOSTOR", "host": "CREWMATE"}))
        self.assertEqual("finished", match.status)
        self.assertEqual("IMPOSTOR", match.roles[self.b])
        self.assertEqual([match.match_id], self.finished)
        self.assertEqual("lobby", self.lobby.state)

    async def test_every_game_starts_with_a_fresh_memory(self) -> None:
        self.registry.handle(self.lobby, GameStarted("skeld", (self.a, self.b)))
        first = self.lobby.match.match_id
        self.registry.handle(self.lobby, ObservationUpdate(observation(self.a, events=(body_found(self.a),), legal=())))
        await self.registry.drain()
        self.registry.handle(self.lobby, GameEnded("crew", "CrewmatesByTask", {}))
        self.registry.handle(self.lobby, GameStarted("skeld", (self.a, self.b)))
        second = self.lobby.match.match_id
        self.assertNotEqual(first, second)
        self.assertEqual([first, second], self.lobby.match_ids)
        self.assertNotIn(session_key(second, self.a), self.runner.sessions)

    async def test_messages_outside_a_game_or_for_strangers_are_refused(self) -> None:
        with self.assertRaises(LobbyError):
            self.registry.handle(self.lobby, DecisionRequest("r1", 30, observation(self.a)))
        self.registry.handle(self.lobby, GameStarted("skeld", (self.a,)))
        with self.assertRaises(LobbyError):
            self.registry.handle(self.lobby, DecisionRequest("r1", 30, observation("mallory")))

    async def test_an_agent_that_left_is_marked(self) -> None:
        self.registry.handle(self.lobby, AgentLeft(self.b, "removed"))
        self.assertFalse(self.lobby.agents[self.b].present)

    async def test_an_action_on_the_agents_own_initiative_reaches_the_server(self) -> None:
        self.registry.handle(self.lobby, GameStarted("skeld", (self.a, self.b)))
        self.registry.push_action(self.lobby, SemanticAction(self.a, ActionKind.CALL_MEETING, goal="ACCUSE"))
        [message] = outbox(self.lobby)
        self.assertIsNone(message.request_id)
        self.assertEqual(ActionKind.CALL_MEETING, message.action.action)
        self.registry.handle(self.lobby, ActionResult(self.a, False, "button used", None))
        entry = self.lobby.match.entries[-1]
        self.assertEqual(("initiative", False, "button used"), (entry["type"], entry["accepted"], entry["rejection"]))
        self.assertEqual({"sent": 1, "refused": 1}, self.lobby.agents[self.a].initiatives)
        with self.assertRaises(LobbyError):
            self.registry.push_action(self.lobby, SemanticAction("mallory", ActionKind.CALL_MEETING))

    async def test_closing_ends_the_running_match(self) -> None:
        self.registry.handle(self.lobby, GameStarted("skeld", (self.a, self.b)))
        match = self.lobby.match
        self.registry.close(self.lobby, "test")
        self.assertEqual("failed", match.status)
        with self.assertRaises(LobbyError):
            self.registry.handle(self.lobby, GameStarted("skeld", (self.a,)))


class InitiativeTests(unittest.IsolatedAsyncioTestCase):
    """Between decisions, an agent whose suspicion reaches its ego's threshold calls a meeting on its own."""

    def _belief(self, suspicion: float, *ids: str) -> dict[str, dict[str, Any]]:
        return {"hawk": {"suspicion": suspicion, "evidence_ids": list(ids)}, "nova": {"suspicion": 0.2, "evidence_ids": []}}

    def _recalled(self, *ids: str, kind: str = "VENT_SEEN", target: str | None = None) -> list[dict[str, Any]]:
        return [{"event_id": i, "kind": kind, "tick": 1, "details": {"actor": "hawk", "target": target}, "certainty": 0.9} for i in ids]

    def _observation(self, *, role: Role = Role.CREWMATE, phase: Phase = Phase.TASKS, meeting: bool = True) -> Observation:
        legal = (LegalAction(ActionKind.MOVE_TO, "admin"), LegalAction(ActionKind.WAIT))
        if meeting:
            legal += (LegalAction(ActionKind.CALL_MEETING),)
        return Observation(agent_id="sherlock", tick=5, phase=phase, self_state=SelfState("sherlock", "admin", role),
                           legal_actions=legal, decision_reason="observed")

    def test_strong_suspicion_calls_a_meeting_citing_its_evidence(self) -> None:
        action, acted, why = initiative_action(self._observation(), self._belief(0.9, "e1"), self._recalled("e1"), load_ego("sherlock"), {})
        self.assertEqual((ActionKind.CALL_MEETING, ("e1",)), (action.action, action.evidence_ids))
        self.assertEqual({"hawk": ["e1"]}, acted)
        self.assertEqual("hawk", why["suspect"])

    def test_the_same_evidence_never_calls_twice_but_new_evidence_does(self) -> None:
        ego = load_ego("sherlock")
        _, acted, _ = initiative_action(self._observation(), self._belief(0.9, "e1"), self._recalled("e1"), ego, {})
        self.assertIsNone(initiative_action(self._observation(), self._belief(0.9, "e1"), self._recalled("e1"), ego, acted)[0])
        again, _, _ = initiative_action(self._observation(), self._belief(0.95, "e1", "e2"), self._recalled("e1", "e2"), ego, acted)
        self.assertIsNotNone(again)

    def test_the_threshold_is_the_egos(self) -> None:
        belief, recalled = self._belief(0.6, "e1"), self._recalled("e1")
        self.assertIsNotNone(initiative_action(self._observation(), belief, recalled, load_ego("maverick"), {})[0])
        self.assertIsNone(initiative_action(self._observation(), belief, recalled, load_ego("echo"), {})[0])

    def test_only_a_crewmate_in_the_task_phase_with_the_button_available(self) -> None:
        belief, recalled, ego = self._belief(1.0, "e1"), self._recalled("e1"), load_ego("maverick")
        for observation in (self._observation(meeting=False), self._observation(role=Role.IMPOSTOR),
                            self._observation(phase=Phase.MEETING)):
            self.assertIsNone(initiative_action(observation, belief, recalled, ego, {})[0])

    def test_nobody_calls_a_meeting_about_the_dead(self) -> None:
        recalled = self._recalled("e1") + [{"event_id": "e9", "kind": "EJECTED", "tick": 3, "details": {"target": "hawk"}, "certainty": 1.0}]
        self.assertIsNone(initiative_action(self._observation(), self._belief(1.0, "e1"), recalled, load_ego("maverick"), {})[0])

    async def test_an_observed_kill_makes_the_agent_call_a_meeting_on_its_own(self) -> None:
        registry = LobbyRegistry(PipelineAgentRunner(), offline=True)
        registered = registry.register(registration(15, players=()))
        lobby = registry.lobby(registered.lobby_id)
        registry.handle(lobby, GameStarted("skeld", tuple(a.agent_id for a in registered.agents)))
        kill = ObservedEvent("maverick-00001", 1, EventKind.KILL_WITNESSED, room="admin", actor="hawk", target="nova")
        seen = Observation(
            agent_id="maverick", tick=2, phase=Phase.TASKS, self_state=SelfState("maverick", "admin", Role.CREWMATE),
            events=(kill,), legal_actions=(LegalAction(ActionKind.CALL_MEETING), LegalAction(ActionKind.WAIT)),
            decision_reason="observed", roster=tuple(a.agent_id for a in registered.agents),
        )
        registry.handle(lobby, ObservationUpdate(seen))
        await registry.drain()
        [message] = outbox(lobby)
        self.assertIsNone(message.request_id)
        self.assertEqual((ActionKind.CALL_MEETING, ("maverick-00001",)), (message.action.action, message.action.evidence_ids))
        [entry] = [e for e in lobby.match.entries if e["type"] == "initiative"]
        self.assertEqual("hawk", entry["reason"]["suspect"])
        registry.handle(lobby, ActionResult("maverick", True, "", None))
        self.assertTrue(entry["accepted"])
        self.assertEqual({"sent": 1, "accepted": 1}, lobby.agents["maverick"].initiatives)

        # The same observation again: the evidence was acted on already.
        registry.handle(lobby, ObservationUpdate(Observation(**{**seen.__dict__, "tick": 3, "events": ()})))
        await registry.drain()
        self.assertEqual([], outbox(lobby))


class FakeBackend:
    """An nlght provider backend as far as the catalog uses it."""

    def __init__(self, *names: str, fail: bool = False) -> None:
        self.names, self.fail, self.calls = names, fail, 0

    async def list_models(self) -> list[Any]:
        self.calls += 1
        if self.fail:
            raise ConnectionError("provider down")
        return [SimpleNamespace(name=n) for n in self.names]


def catalog() -> ModelCatalog:
    return ModelCatalog(
        {"ollama": FakeBackend("qwen3-coder:30b", "llama3.2:3b"), "anthropic": FakeBackend("claude-sonnet-5-5"), "down": FakeBackend(fail=True)},
        {"ollama": "qwen3-coder:30b", "anthropic": "claude-sonnet-5-5", "down": None},
    )


class RecordingRunner(PipelineAgentRunner):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict[str, Any]] = []

    async def decide(self, *, agent_id: str, session_key: str, payload: dict[str, Any], mode: str = "ego") -> dict[str, Any]:
        self.payloads.append(payload)
        return await super().decide(agent_id=agent_id, session_key=session_key, payload=payload, mode=mode)


class ModelTests(unittest.IsolatedAsyncioTestCase):
    """Each agent decides with its own model: the operator's choice, else its ego's, else the platform default."""

    async def test_the_catalog_lists_what_each_provider_serves(self) -> None:
        models = catalog()
        listing = {p["name"]: p for p in await models.listing()}
        self.assertEqual(["llama3.2:3b", "qwen3-coder:30b"], listing["ollama"]["models"])
        self.assertEqual("provider down", listing["down"]["error"])
        await models.listing()
        self.assertEqual(1, models._backends["ollama"].calls, "the list is reused for a while")

    async def test_a_choice_is_checked_against_the_provider(self) -> None:
        models = catalog()
        self.assertEqual(ModelChoice("anthropic", "claude-sonnet-5-5", "override"), await models.validate("anthropic", None))
        with self.assertRaises(ModelChoiceError):
            await models.validate("openai", "gpt")
        with self.assertRaises(ModelChoiceError):
            await models.validate("ollama", "no-such-model")
        # A provider that cannot list its models right now is not second-guessed.
        self.assertEqual("x", (await models.validate("down", "x")).model)

    def test_resolution_order(self) -> None:
        models = catalog()
        plain = load_ego("sherlock")
        self.assertEqual(ModelChoice("ollama", "qwen3-coder:30b", "default"), models.resolve(plain))
        named = parse_ego("sherlock", Path("among_us/egos/sherlock.toml").read_text() + '\n[model]\nprovider = "anthropic"\n')
        self.assertEqual(ModelChoice("anthropic", "claude-sonnet-5-5", "ego"), models.resolve(named))
        override = ModelChoice("ollama", "llama3.2:3b", "override")
        self.assertEqual(override, models.resolve(named, override))
        self.assertFalse(models.usable(ModelChoice("openai", None, "ego")))

    def test_an_ego_file_names_its_model_strictly(self) -> None:
        text = Path("among_us/egos/sherlock.toml").read_text()
        self.assertEqual("llama3.2:3b", parse_ego("sherlock", text + '\n[model]\nprovider = "ollama"\nname = "llama3.2:3b"\n').model.name)
        with self.assertRaises(InvalidEgo):
            parse_ego("sherlock", text + '\n[model]\nprovider = "ollama"\ntemperature = 0.2\n')

    async def test_every_turn_names_its_agents_provider_and_model(self) -> None:
        runner = RecordingRunner()
        registry = LobbyRegistry(runner, catalog=catalog())
        registered = registry.register(registration(2))
        lobby = registry.lobby(registered.lobby_id)
        a, b = (agent.agent_id for agent in registered.agents)
        registry.set_model(lobby, b, await registry.catalog.validate("anthropic", None))
        registry.handle(lobby, GameStarted("skeld", (a, b)))
        registry.handle(lobby, DecisionRequest("r1", 30, observation(a)))
        registry.handle(lobby, DecisionRequest("r2", 30, observation(b)))
        await registry.drain()
        used = {p["agent_id"]: (p["model_provider"], p["model"]) for p in runner.payloads}
        self.assertEqual(("ollama", "qwen3-coder:30b"), used[a])
        self.assertEqual(("anthropic", "claude-sonnet-5-5"), used[b])
        self.assertEqual("override", lobby.agents[b].summary()["model"]["source"])

        registry.set_model(lobby, b, None)
        self.assertEqual("default", lobby.agents[b].model.source)

    def test_an_ego_may_name_a_model_for_meetings(self) -> None:
        models = catalog()
        text = Path("among_us/egos/sherlock.toml").read_text()
        both = parse_ego("sherlock", text + '\n[model]\nprovider = "ollama"\nname = "llama3.2:3b"\n\n[model.meeting]\nprovider = "anthropic"\n')
        self.assertEqual(ModelChoice("ollama", "llama3.2:3b", "ego"), models.resolve(both))
        self.assertEqual(ModelChoice("anthropic", "claude-sonnet-5-5", "ego"), models.resolve(both, meeting=True))

    async def test_meeting_turns_run_on_the_meeting_model(self) -> None:
        runner = RecordingRunner()
        registry = LobbyRegistry(runner, catalog=catalog())
        registered = registry.register(registration(1))
        lobby = registry.lobby(registered.lobby_id)
        agent = registered.agents[0].agent_id
        self.assertEqual(lobby.agents[agent].model, lobby.agents[agent].meeting_model, "without a meeting model, the game model")
        registry.set_model(lobby, agent, await registry.catalog.validate("anthropic", None), meeting=True)
        registry.handle(lobby, GameStarted("skeld", (agent,)))
        registry.handle(lobby, DecisionRequest("r1", 120, observation(agent)))
        await registry.drain()   # a newer request of the same agent would replace this one
        meeting = Observation(agent_id=agent, tick=3, phase=Phase.MEETING, self_state=SelfState(agent, "admin", Role.CREWMATE),
                              legal_actions=(LegalAction(ActionKind.SAY), LegalAction(ActionKind.WAIT)), roster=(agent,))
        registry.handle(lobby, DecisionRequest("r2", 120, meeting))
        await registry.drain()
        used = {p["observation"]["phase"]: p["model_provider"] for p in runner.payloads}
        self.assertEqual({"TASKS": "ollama", "MEETING": "anthropic"}, used)

    async def test_an_agent_whose_provider_is_missing_decides_by_policy_and_says_so(self) -> None:
        registry = LobbyRegistry(RecordingRunner(), catalog=catalog())
        registered = registry.register(registration(1))
        lobby = registry.lobby(registered.lobby_id)
        agent = registered.agents[0].agent_id
        registry.set_model(lobby, agent, ModelChoice("openai", "gpt", "override"))
        self.assertFalse(lobby.agents[agent].summary()["model_usable"])
        registry.handle(lobby, GameStarted("skeld", (agent,)))
        registry.handle(lobby, DecisionRequest("r1", 30, observation(agent)))
        await registry.drain()
        self.assertEqual({"policy:no-model": 1}, dict(lobby.agents[agent].decided))

    async def test_the_seed_must_not_pin_a_provider(self) -> None:
        def container(config: dict[str, Any]) -> Any:
            step = SimpleNamespace(type="among_us_decide", config=config)

            class Repository:
                async def find_by_name(self, name: str) -> Any:
                    return SimpleNamespace(workflow_id=name, enabled=True)

                async def find_active_version(self, workflow_id: Any) -> Any:
                    return SimpleNamespace(steps=[step])

            return SimpleNamespace(workflow_repository=Repository())

        self.assertIsNone(await _seed_problem(container({"debug": False})))
        self.assertIn("load seed/08_among_us.sql again", await _seed_problem(container({"model_provider": "ollama"})))


class AdminViewTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_lobby_shows_deaths_beliefs_and_memories(self) -> None:
        registry = LobbyRegistry(PipelineAgentRunner(), offline=True)
        registered = registry.register(registration(15, players=()))
        lobby = registry.lobby(registered.lobby_id)
        # A sharp observer: what a weak ego keeps of a kill may stay below what it can recall.
        a, b = "sherlock", "echo"
        registry.handle(lobby, GameStarted("skeld", (a, b, "host")))
        kill = ObservedEvent(f"{a}-00001", 1, EventKind.KILL_WITNESSED, room="admin", actor="host", target=b)
        registry.handle(lobby, ObservationUpdate(observation(a, events=(kill,), legal=())))
        self.assertFalse(lobby.agents[b].alive, "the victim is dead as soon as anyone saw it")
        await registry.drain()
        seen = lobby.agents[a].summary()
        self.assertIn(f"{a}-00001", [m["event_id"] for m in seen["recalled"]])
        self.assertGreater(seen["beliefs"]["host"]["suspicion"], 0.2)
        self.assertEqual(1, seen["memory"]["stored"])


class ReplacedRequestTests(unittest.IsolatedAsyncioTestCase):
    async def _lobby(self, runner: PipelineAgentRunner):
        registry = LobbyRegistry(runner)
        registered = registry.register(registration(1))
        lobby = registry.lobby(registered.lobby_id)
        registry.handle(lobby, GameStarted("skeld", (registered.agents[0].agent_id,)))
        return registry, lobby, registered.agents[0].agent_id

    async def test_a_replaced_request_is_not_answered(self) -> None:
        runner = SlowModelRunner(delay=0.3)
        registry, lobby, agent = await self._lobby(runner)
        registry.handle(lobby, DecisionRequest("r1", 30, observation(agent, tick=2)))
        await asyncio.sleep(0.05)
        registry.handle(lobby, DecisionRequest("r2", 30, observation(agent, tick=3)))
        await registry.drain()
        self.assertEqual(["r2"], [m.request_id for m in outbox(lobby)])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LobbyHttpTests(unittest.IsolatedAsyncioTestCase):
    """The lobby endpoints as the nlght Impostor server uses them, over real HTTP."""

    async def asyncSetUp(self) -> None:
        self.registry = LobbyRegistry(PipelineAgentRunner(), offline=True, catalog=catalog())
        app = FastAPI()
        app.include_router(AmongUsAdapter(run_dir=self._tmp(), registry=self.registry).build_router())
        self.port = _free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.serving = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.02)

    async def asyncTearDown(self) -> None:
        self.server.should_exit = True
        await self.serving

    def _tmp(self) -> str:
        import tempfile

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return directory.name

    async def _request(self, method: str, path: str, body: Any = None, token: str | None = None) -> tuple[int, Any]:
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        data = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        headers = [f"{method} {path} HTTP/1.1", f"Host: 127.0.0.1:{self.port}", "Connection: close",
                   f"Content-Length: {len(data)}", "Content-Type: application/json"]
        if token:
            headers.append(f"Authorization: Bearer {token}")
        writer.write(("\r\n".join(headers) + "\r\n\r\n").encode() + data)
        await writer.drain()
        raw = await reader.read()
        writer.close()
        head, _, payload = raw.partition(b"\r\n\r\n")
        status = int(head.split(b" ")[1])
        return status, json.loads(payload) if payload.strip().startswith((b"{", b"[")) else payload

    async def _events(self, lobby_id: str, token: str, count: int) -> list[Any]:
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write((f"GET /among-us/api/lobbies/{lobby_id}/stream HTTP/1.1\r\nHost: x\r\n"
                      f"Authorization: Bearer {token}\r\n\r\n").encode())
        await writer.drain()
        messages = []
        try:
            while len(messages) < count:
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                text = line.decode().strip()
                if text.startswith("data: {\"type\""):
                    messages.append(decode(text.removeprefix("data: ")))
        finally:
            writer.close()
        return messages

    async def test_register_play_and_answer_over_http(self) -> None:
        raw = json.loads((FIXTURES / "lobby_registration.json").read_text(encoding="utf-8"))
        status, registered = await self._request("POST", "/among-us/api/lobbies", raw)
        self.assertEqual(201, status)
        self.assertEqual(2, registered["protocol"])
        lobby_id, token = registered["lobby_id"], registered["token"]
        agent = registered["agents"][0]["agent_id"]

        post = lambda message: self._request("POST", f"/among-us/api/lobbies/{lobby_id}/messages",
                                             encode(message).encode(), token)
        self.assertEqual(202, (await post(GameStarted("skeld", (agent, "host"))))[0])
        events = asyncio.create_task(self._events(lobby_id, token, 1))
        self.assertEqual(202, (await post(DecisionRequest("r7", 30, observation(agent))))[0])
        [answer] = await events
        self.assertEqual(("r7", agent), (answer.request_id, answer.action.agent_id))

        status, listing = await self._request("GET", "/among-us/api/lobbies")
        self.assertEqual(200, status)
        [entry] = listing["lobbies"]
        self.assertEqual("in_game", entry["state"])
        self.assertNotIn("token", json.dumps(entry))

        status, more = await self._request("POST", f"/among-us/api/lobbies/{lobby_id}/agents",
                                           {"protocol": 2, "count": 1, "players": ["Host"]}, token)
        self.assertEqual(201, status)
        self.assertEqual(1, len(more["agents"]))
        self.assertEqual(204, (await self._request("DELETE", f"/among-us/api/lobbies/{lobby_id}", token=token))[0])

    async def test_the_operator_chooses_an_agents_model(self) -> None:
        status, models = await self._request("GET", "/among-us/api/models")
        self.assertEqual(200, status)
        self.assertEqual(["ollama", "anthropic", "down"], [p["name"] for p in models["providers"]])
        self.assertEqual("ollama", models["default"]["provider"])

        raw = json.loads((FIXTURES / "lobby_registration.json").read_text(encoding="utf-8"))
        _, registered = await self._request("POST", "/among-us/api/lobbies", raw)
        path = f"/among-us/api/lobbies/{registered['lobby_id']}/agents/{registered['agents'][0]['agent_id']}/model"
        status, agent = await self._request("PUT", path, {"provider": "ollama", "model": "llama3.2:3b"})
        self.assertEqual(200, status)
        self.assertEqual({"provider": "ollama", "model": "llama3.2:3b", "source": "override"}, agent["model"])
        self.assertEqual(422, (await self._request("PUT", path, {"provider": "ollama", "model": "gpt-9"}))[0])
        status, agent = await self._request("PUT", path, {"provider": None})
        self.assertEqual("default", agent["model"]["source"])

    async def test_a_lobby_answers_only_its_own_token(self) -> None:
        raw = json.loads((FIXTURES / "lobby_registration.json").read_text(encoding="utf-8"))
        _, first = await self._request("POST", "/among-us/api/lobbies", raw)
        _, second = await self._request("POST", "/among-us/api/lobbies", raw)
        path = f"/among-us/api/lobbies/{first['lobby_id']}/messages"
        message = encode(GameStarted("skeld", ())).encode()
        self.assertEqual(401, (await self._request("POST", path, message))[0])
        self.assertEqual(403, (await self._request("POST", path, message, second["token"]))[0])
        self.assertEqual(403, (await self._request("DELETE", f"/among-us/api/lobbies/{first['lobby_id']}", token=second["token"]))[0])
        self.assertEqual(202, (await self._request("POST", path, message, first["token"]))[0])

    async def test_a_malformed_message_is_refused(self) -> None:
        raw = json.loads((FIXTURES / "lobby_registration.json").read_text(encoding="utf-8"))
        _, registered = await self._request("POST", "/among-us/api/lobbies", raw)
        leaky = (FIXTURES / "invalid_leaky_observation.json").read_bytes()
        status, _ = await self._request("POST", f"/among-us/api/lobbies/{registered['lobby_id']}/messages",
                                        leaky, registered["token"])
        self.assertEqual(422, status)

    async def test_the_registry_is_not_created_without_the_seeded_workflow(self) -> None:
        from types import SimpleNamespace

        class NoWorkflows:
            async def find_by_name(self, name: str) -> None:
                return None

        app = FastAPI()
        app.state.container = SimpleNamespace(workflow_repository=NoWorkflows())
        app.include_router(AmongUsAdapter(run_dir=self._tmp()).build_router())
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        serving = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.02)
        try:
            self.port, own = port, self.port
            raw = json.loads((FIXTURES / "lobby_registration.json").read_text(encoding="utf-8"))
            status, body = await self._request("POST", "/among-us/api/lobbies", raw)
            self.assertEqual(503, status)
            self.assertIn("08_among_us.sql", json.dumps(body))
            self.port = own
        finally:
            server.should_exit = True
            await serving


if __name__ == "__main__":
    unittest.main()
