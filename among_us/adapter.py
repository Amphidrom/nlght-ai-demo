"""HTTP surface for starting and watching Among Us matches.

POST /among-us/api/matches                 start a simulated match
POST /among-us/api/lobbies                 the nlght Impostor server registers a lobby (``!bots``)
POST /among-us/api/lobbies/{id}/agents     … seats more agents in it            (lobby token)
POST /among-us/api/lobbies/{id}/messages   … posts a game message               (lobby token)
GET  /among-us/api/lobbies/{id}/stream     … receives actions as server-sent events (lobby token)
DELETE /among-us/api/lobbies/{id}          … closes it                          (lobby token)
GET  /among-us/api/lobbies                 all lobbies, their agents and matches
GET  /among-us/api/lobbies/{id}            one lobby
PUT  /among-us/api/lobbies/{id}/agents/{agent}/model   the operator's model for one agent's game or meeting turns
                                           ({"provider", "model", "phase": "game"|"meeting"}; provider null: back to default)
GET  /among-us/api/models                  configured providers and the models each one serves
GET  /among-us/api/matches/{id}            status and result
GET  /among-us/api/matches/{id}/events     decision log as server-sent events
GET  /among-us/api/matches/{id}/evaluation metrics against the ground truth
GET  /among-us/api/matches                  all matches of this process
GET  /among-us/api/matches/{id}/spectator  ground truth for the human spectator (simulation)
GET  /among-us/api/egos                    available cognitive profiles
GET  /among-us/api/map                     rooms and corridors of the simulated ship
GET  /among-us                             spectator dashboard
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from among_us.ego import ego_ids, load_ego
from among_us.environment import GameEnvironment, SimEnvironment
from among_us.evaluation import evaluate
from among_us.lobby import Lobby, LobbyError, LobbyRegistry
from among_us.models import ModelCatalog, ModelChoiceError, catalog_from_container
from among_us.match import (
    MEMORY_MODES,
    MatchRecord,
    PipelineAgentRunner,
    ReplayWriter,
    WorkflowAgentRunner,
    run_match,
)
from among_us.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode,
    registered_to_dict,
    registration_from_dict,
    to_dict,
)
from among_us.scheduler import DecisionScheduler, SchedulerConfig
from among_us.sim.engine import MAX_PLAYERS, MIN_PLAYERS, SPECIAL_ROLES, max_impostors
from among_us.sim.map import CORRIDORS, ROOMS

logger = logging.getLogger(__name__)
_TERMINAL = {"finished", "failed"}
_STATIC = Path(__file__).with_name("static")
#: The only files the dashboard route serves.
_ASSETS = {"index.html": "text/html", "styles.css": "text/css", "app.js": "text/javascript"}


class AmongUsAdapter:
    def __init__(self, *, run_dir: str = ".among-us-runs", registry: LobbyRegistry | None = None) -> None:
        self._matches: dict[str, MatchRecord] = {}
        self._environments: dict[str, GameEnvironment] = {}
        self._replays = ReplayWriter(run_dir)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._registry = registry
        self._catalog: ModelCatalog | None = None
        if registry is not None:
            self._attach(registry)

    def _attach(self, registry: LobbyRegistry) -> None:
        registry._on_started = lambda record: self._matches.__setitem__(record.match_id, record)
        registry._on_finished = self._lobby_match_finished

    def _lobby_match_finished(self, record: MatchRecord) -> None:
        path = self._replays.write(record)
        logger.info("among_us.match.%s | match=%s | replay=%s", record.status, record.match_id, path)

    async def _lobbies(self, request: Request) -> LobbyRegistry:
        """The registry, created on first use: agents decide through the seeded workflow, never silently without."""

        if self._registry is None:
            container = request.app.state.container
            problem = await _seed_problem(container)
            if problem:
                raise HTTPException(503, problem)
            catalog = self._models(request)
            self._registry = LobbyRegistry(WorkflowAgentRunner(container, catalog=catalog), catalog=catalog)
            self._attach(self._registry)
        return self._registry

    def _models(self, request: Request) -> ModelCatalog:
        if self._catalog is None:
            self._catalog = catalog_from_container(request.app.state.container)
        return self._catalog

    def build_router(self) -> APIRouter:
        router = APIRouter(tags=["among-us"])

        @router.get("/among-us", include_in_schema=False)
        async def dashboard() -> HTMLResponse:
            return HTMLResponse(_asset("index.html"))

        @router.get("/among-us/assets/{name}", include_in_schema=False)
        async def asset(name: str) -> Response:
            media = _ASSETS.get(name)
            if media is None:
                raise HTTPException(404, "Asset not found.")
            return Response(_asset(name), media_type=media)

        @router.get("/among-us/api/map")
        async def ship_map() -> dict[str, Any]:
            """Rooms and corridors of the simulated ship, for drawing it."""
            return {"rooms": list(ROOMS), "corridors": [list(edge) for edge in CORRIDORS]}

        @router.get("/among-us/api/egos")
        async def egos() -> dict[str, Any]:
            return {"egos": [{"id": e, "name": load_ego(e).name} for e in ego_ids()]}

        @router.post("/among-us/api/matches", status_code=202)
        async def start(request: Request) -> JSONResponse:
            body = await _json_body(request)
            options = _match_options(body)
            container = request.app.state.container
            if not options["offline"]:
                problem = await _seed_problem(container)
                if problem:
                    raise HTTPException(503, problem)
            match_id = uuid.uuid4().hex[:12]
            record = MatchRecord(match_id, options["seed"], options["assignment"], options["environment"],
                                 modes=options["modes"])
            self._matches[match_id] = record
            environment = self._environment(options)
            self._environments[match_id] = environment
            runner = PipelineAgentRunner() if options["offline"] else WorkflowAgentRunner(
                container, catalog=self._models(request),
            )
            scheduler = DecisionScheduler(options["scheduler"], model_enabled=not options["offline"])
            task = asyncio.create_task(self._run(environment, runner, scheduler, record))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return JSONResponse(record.summary(), status_code=202)

        @router.post("/among-us/api/lobbies", status_code=201)
        async def register(request: Request) -> dict[str, Any]:
            registry = await self._lobbies(request)
            try:
                registration = registration_from_dict(await _json_body(request))
                return registered_to_dict(registry.register(registration))
            except ProtocolError as exc:
                raise HTTPException(422, str(exc)) from exc
            except LobbyError as exc:
                raise HTTPException(exc.status, str(exc)) from exc

        @router.post("/among-us/api/lobbies/{lobby_id}/agents", status_code=201)
        async def add_agents(lobby_id: str, request: Request) -> dict[str, Any]:
            registry = await self._lobbies(request)
            lobby = _authorized(registry, lobby_id, request)
            body = await _json_body(request)
            if body.pop("protocol", None) != PROTOCOL_VERSION or set(body) - {"count", "players"}:
                raise HTTPException(422, "Expected {protocol, count, players}.")
            count, players = body.get("count"), body.get("players") or []
            if not isinstance(count, int) or isinstance(count, bool) or not isinstance(players, list) \
                    or not all(isinstance(p, str) for p in players):
                raise HTTPException(422, "count must be an integer and players a list of names.")
            try:
                seated = registry.add_agents(lobby, count, tuple(players))
            except LobbyError as exc:
                raise HTTPException(exc.status, str(exc)) from exc
            return {"protocol": PROTOCOL_VERSION, "agents": [to_dict(seat) for seat in seated]}

        @router.post("/among-us/api/lobbies/{lobby_id}/messages", status_code=202)
        async def message(lobby_id: str, request: Request) -> Response:
            registry = await self._lobbies(request)
            lobby = _authorized(registry, lobby_id, request)
            try:
                registry.handle(lobby, decode(await request.body()))
            except ProtocolError as exc:
                logger.error("among_us.lobby.protocol_error | lobby=%s | %s", lobby_id, exc)
                raise HTTPException(422, str(exc)) from exc
            except LobbyError as exc:
                raise HTTPException(exc.status, str(exc)) from exc
            return Response(status_code=202)

        @router.get("/among-us/api/lobbies/{lobby_id}/stream")
        async def stream(lobby_id: str, request: Request) -> StreamingResponse:
            registry = await self._lobbies(request)
            lobby = _authorized(registry, lobby_id, request)
            return StreamingResponse(_lobby_stream(lobby), media_type="text/event-stream")

        @router.delete("/among-us/api/lobbies/{lobby_id}", status_code=204)
        async def close(lobby_id: str, request: Request) -> Response:
            registry = await self._lobbies(request)
            registry.close(_authorized(registry, lobby_id, request), "closed by the server")
            return Response(status_code=204)

        @router.get("/among-us/api/lobbies")
        async def lobbies() -> dict[str, Any]:
            registry = self._registry
            items = [] if registry is None else sorted(registry.lobbies.values(), key=lambda lobby: -lobby.created_at)
            return {"lobbies": [lobby.summary() for lobby in items]}

        @router.get("/among-us/api/lobbies/{lobby_id}")
        async def lobby(lobby_id: str) -> dict[str, Any]:
            if self._registry is None:
                raise HTTPException(404, "Lobby not found.")
            try:
                return self._registry.lobby(lobby_id).summary()
            except LobbyError as exc:
                raise HTTPException(exc.status, str(exc)) from exc

        @router.get("/among-us/api/models")
        async def models(request: Request) -> dict[str, Any]:
            """Every configured provider with the models it serves right now, asked through its nlght backend."""

            catalog = self._registry.catalog if self._registry is not None and self._registry.catalog else self._models(request)
            default = catalog.default()
            return {"providers": await catalog.listing(), "default": default.to_dict() if default else None}

        @router.put("/among-us/api/lobbies/{lobby_id}/agents/{agent_id}/model")
        async def agent_model(lobby_id: str, agent_id: str, request: Request) -> dict[str, Any]:
            if self._registry is None:
                raise HTTPException(404, "Lobby not found.")
            body = await _json_body(request)
            provider, model, phase = body.get("provider"), body.get("model"), body.get("phase", "game")
            if (set(body) - {"provider", "model", "phase"} or not isinstance(provider, str | None)
                    or not isinstance(model, str | None) or phase not in ("game", "meeting")):
                raise HTTPException(422, "Expected {provider, model, phase}: strings (phase game or meeting), "
                                         "or provider null to go back to the default.")
            try:
                lobby = self._registry.lobby(lobby_id)
                choice = None
                if provider is not None:
                    catalog = self._registry.catalog
                    if catalog is None:
                        raise HTTPException(409, "No model providers are known in this process.")
                    choice = await catalog.validate(provider, model or None)
                return self._registry.set_model(lobby, agent_id, choice, meeting=phase == "meeting").summary()
            except ModelChoiceError as exc:
                raise HTTPException(422, str(exc)) from exc
            except LobbyError as exc:
                raise HTTPException(exc.status, str(exc)) from exc

        @router.get("/among-us/api/matches/{match_id}")
        async def status(match_id: str) -> dict[str, Any]:
            return self._require(match_id).summary()

        @router.get("/among-us/api/matches/{match_id}/events")
        async def events(match_id: str) -> StreamingResponse:
            record = self._require(match_id)
            return StreamingResponse(self._stream(record), media_type="text/event-stream")

        @router.get("/among-us/api/matches/{match_id}/spectator")
        async def spectator(match_id: str) -> dict[str, Any]:
            """Ground truth for the human spectator (simulation only); agents never see it."""
            self._require(match_id)
            view = getattr(self._environments.get(match_id), "spectator_view", None)
            if view is None:
                raise HTTPException(404, "No spectator view for this environment.")
            return view()

        @router.get("/among-us/api/matches")
        async def matches() -> dict[str, Any]:
            return {"matches": [record.summary() for record in reversed(self._matches.values())]}

        @router.get("/among-us/api/matches/{match_id}/evaluation")
        async def evaluation(match_id: str) -> dict[str, Any]:
            record = self._require(match_id)
            if record.status not in _TERMINAL:
                raise HTTPException(409, "The match is still running.")
            return evaluate(record)

        return router

    async def _run(self, environment: GameEnvironment, runner: Any, scheduler: DecisionScheduler, record: MatchRecord) -> None:
        await run_match(environment, runner, record, scheduler=scheduler)
        path = self._replays.write(record)
        logger.info("among_us.match.%s | match=%s | replay=%s", record.status, record.match_id, path)

    @staticmethod
    def _environment(options: dict[str, Any]) -> GameEnvironment:
        return SimEnvironment(
            list(options["assignment"]), seed=options["seed"], impostors=options["impostors"],
            impostor_roles=options["impostor_roles"],
        )

    def _require(self, match_id: str) -> MatchRecord:
        record = self._matches.get(match_id)
        if record is None:
            raise HTTPException(404, "Match not found.")
        return record

    async def _stream(self, record: MatchRecord) -> AsyncIterator[str]:
        cursor = 0
        while True:
            while cursor < len(record.entries):
                entry = record.entries[cursor]
                cursor += 1
                yield f"event: {entry['type']}\ndata: {json.dumps(entry, ensure_ascii=False)}\n\n"
            if record.status in _TERMINAL:
                yield f"event: done\ndata: {json.dumps(record.summary())}\n\n"
                return
            yield ": keep-alive\n\n"
            await record.wait_for_change()


def _authorized(registry: LobbyRegistry, lobby_id: str, request: Request) -> Lobby:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "A lobby token is required.")
    try:
        return registry.lobby(lobby_id, token)
    except LobbyError as exc:
        raise HTTPException(exc.status, str(exc)) from exc


async def _lobby_stream(lobby: Lobby) -> AsyncIterator[str]:
    """Actions for the server, one protocol message per event, until the lobby closes."""

    while lobby.state != "closed":
        try:
            line = await asyncio.wait_for(lobby.outbox.get(), timeout=15)
        except TimeoutError:
            yield ": keep-alive\n\n"
            continue
        yield f"data: {line}\n\n"
    yield "event: closed\ndata: {}\n\n"


def _asset(name: str) -> str:
    return (_STATIC / name).read_text(encoding="utf-8")


def _match_options(body: dict[str, Any]) -> dict[str, Any]:
    environment = str(body.get("environment", "sim"))
    if environment == "bridge":
        raise HTTPException(422, "The real game is not started here: the nlght Impostor server registers each lobby "
                                 "when its host types !bots (POST /among-us/api/lobbies).")
    if environment != "sim":
        raise HTTPException(422, "environment must be 'sim'.")
    # Default: a full lobby, one player per ego.
    egos = body.get("egos") or list(ego_ids())
    if not isinstance(egos, list) or not all(isinstance(e, str) for e in egos):
        raise HTTPException(422, "egos must be a list of ego ids.")
    unknown = sorted(set(egos) - set(ego_ids()))
    if unknown:
        raise HTTPException(422, f"Unknown egos: {unknown}.")
    # Players are named after their ego.
    if len(set(egos)) != len(egos) or not MIN_PLAYERS <= len(egos) <= MAX_PLAYERS:
        raise HTTPException(422, f"A simulated match needs {MIN_PLAYERS} to {MAX_PLAYERS} distinct egos.")
    assignment = {e: e for e in egos}
    try:
        seed = int(body.get("seed", 7))
        impostors = int(body.get("impostors", min(2, max_impostors(len(assignment)))))
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "seed and impostors must be integers.") from exc
    if not 1 <= impostors <= max_impostors(len(assignment)):
        raise HTTPException(422, f"{len(assignment)} players allow 1 to {max_impostors(len(assignment))} impostors.")
    # Players that play with the transcript baseline instead of ego memory.
    baseline = body.get("baseline") or []
    if not isinstance(baseline, list) or not set(baseline) <= set(assignment):
        raise HTTPException(422, "baseline must list players of this match.")
    modes = {player: "baseline" if player in baseline else "ego" for player in assignment}
    try:
        scheduler = SchedulerConfig.model_validate(body.get("scheduler") or {})
    except ValidationError as exc:
        raise HTTPException(422, f"Invalid scheduler settings: {exc.errors(include_url=False)}") from exc
    roles = body.get("impostor_roles") or []
    if not isinstance(roles, list) or set(roles) - set(SPECIAL_ROLES) or len(roles) > impostors:
        raise HTTPException(422, f"impostor_roles must list at most {impostors} of {list(SPECIAL_ROLES)}.")
    return {
        "impostor_roles": roles,
        "scheduler": scheduler,
        "modes": modes,
        "environment": environment,
        "assignment": assignment,
        "seed": seed,
        "impostors": impostors,
        "offline": bool(body.get("offline", False)),
    }


async def _seed_problem(container: Any) -> str | None:
    """Why agent turns cannot run on the seeded workflows, if they cannot."""

    repository = container.workflow_repository
    for name in sorted(set(MEMORY_MODES.values())):
        workflow = await repository.find_by_name(name) if repository else None
        if workflow is None or not workflow.enabled:
            return f"Workflow '{name}' is not seeded; load seed/08_among_us.sql."
        version = await repository.find_active_version(workflow.workflow_id)
        if version and any(step.type == "among_us_decide" and "model_provider" in step.config for step in version.steps):
            # A pinned provider wins over the one each turn names: every agent would think with the same model.
            return (f"Workflow '{name}' pins its model provider, so agents cannot use their own models; "
                    "load seed/08_among_us.sql again.")
    return None


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(400, "Request body must be valid JSON.") from exc
    if not isinstance(body, dict):
        raise HTTPException(422, "Request body must be a JSON object.")
    return body
