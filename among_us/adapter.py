"""HTTP surface for starting and watching Among Us matches.

POST /among-us/api/matches                 start a match (simulation or bridge)
GET  /among-us/api/matches/{id}            status and result
GET  /among-us/api/matches/{id}/events     decision log as server-sent events
GET  /among-us/api/matches/{id}/evaluation metrics against the ground truth
GET  /among-us/api/egos                    available cognitive profiles
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from among_us.ego import ego_ids, load_ego
from among_us.environment import BridgeEnvironment, GameEnvironment, SimEnvironment
from among_us.evaluation import evaluate
from among_us.match import (
    MEMORY_MODES,
    MatchRecord,
    PipelineAgentRunner,
    ReplayWriter,
    WorkflowAgentRunner,
    run_match,
)
from among_us.scheduler import DecisionScheduler, SchedulerConfig
from among_us.sim.engine import MAX_PLAYERS, MIN_PLAYERS, max_impostors

logger = logging.getLogger(__name__)
_TERMINAL = {"finished", "failed"}


class AmongUsAdapter:
    def __init__(self, *, run_dir: str = ".among-us-runs") -> None:
        self._matches: dict[str, MatchRecord] = {}
        self._replays = ReplayWriter(run_dir)
        self._tasks: set[asyncio.Task[Any]] = set()

    def build_router(self) -> APIRouter:
        router = APIRouter(tags=["among-us"])

        @router.get("/among-us/api/egos")
        async def egos() -> dict[str, Any]:
            return {"egos": [{"id": e, "name": load_ego(e).name} for e in ego_ids()]}

        @router.post("/among-us/api/matches", status_code=202)
        async def start(request: Request) -> JSONResponse:
            body = await _json_body(request)
            options = _match_options(body)
            container = request.app.state.container
            if not options["offline"]:
                repository = container.workflow_repository
                for name in {MEMORY_MODES[mode] for mode in options["modes"].values()}:
                    workflow = await repository.find_by_name(name) if repository else None
                    if workflow is None or not workflow.enabled:
                        raise HTTPException(503, f"Workflow '{name}' is not seeded; load seed/08_among_us.sql.")
            match_id = uuid.uuid4().hex[:12]
            record = MatchRecord(match_id, options["seed"], options["assignment"], options["environment"],
                                 modes=options["modes"])
            self._matches[match_id] = record
            environment = self._environment(options)
            runner = PipelineAgentRunner() if options["offline"] else WorkflowAgentRunner(
                container, model=_default_model(container),
            )
            scheduler = DecisionScheduler(options["scheduler"], model_enabled=not options["offline"])
            task = asyncio.create_task(self._run(environment, runner, scheduler, record))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return JSONResponse(record.summary(), status_code=202)

        @router.get("/among-us/api/matches/{match_id}")
        async def status(match_id: str) -> dict[str, Any]:
            return self._require(match_id).summary()

        @router.get("/among-us/api/matches/{match_id}/events")
        async def events(match_id: str) -> StreamingResponse:
            record = self._require(match_id)
            return StreamingResponse(self._stream(record), media_type="text/event-stream")

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
        if options["environment"] == "bridge":
            return BridgeEnvironment(options["bridge_host"], options["bridge_port"])
        return SimEnvironment(list(options["assignment"]), seed=options["seed"], impostors=options["impostors"])

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


def _match_options(body: dict[str, Any]) -> dict[str, Any]:
    environment = str(body.get("environment", "sim"))
    if environment not in ("sim", "bridge"):
        raise HTTPException(422, "environment must be 'sim' or 'bridge'.")
    # Default: a full lobby, one player per ego.
    egos = body.get("egos") or list(ego_ids())
    if not isinstance(egos, list) or not all(isinstance(e, str) for e in egos):
        raise HTTPException(422, "egos must be a list of ego ids.")
    unknown = sorted(set(egos) - set(ego_ids()))
    if unknown:
        raise HTTPException(422, f"Unknown egos: {unknown}.")
    # Players are named after their ego, unless the bridge seats them:
    # then ``players`` maps the bridge's player ids to egos.
    players = body.get("players")
    if environment == "bridge":
        if not isinstance(players, dict) or not players:
            raise HTTPException(422, "A bridge match needs players: {bridge_player_id: ego_id}.")
        assignment = {str(k): str(v) for k, v in players.items()}
        if set(assignment.values()) - set(ego_ids()):
            raise HTTPException(422, "players maps to an unknown ego.")
    else:
        if len(set(egos)) != len(egos) or not MIN_PLAYERS <= len(egos) <= MAX_PLAYERS:
            raise HTTPException(422, f"A simulated match needs {MIN_PLAYERS} to {MAX_PLAYERS} distinct egos.")
        assignment = {e: e for e in egos}
    try:
        seed = int(body.get("seed", 7))
        impostors = int(body.get("impostors", min(2, max_impostors(len(assignment)))))
        port = int(body.get("bridge_port", 47100))
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "seed, impostors and bridge_port must be integers.") from exc
    # The bridge is always local, and only its own port range is reachable:
    # a request cannot point the orchestrator at another local service.
    if not 1 <= impostors <= max_impostors(len(assignment)):
        raise HTTPException(422, f"{len(assignment)} players allow 1 to {max_impostors(len(assignment))} impostors.")
    if not 47100 <= port <= 47199:
        raise HTTPException(422, "bridge_port must be within 47100-47199.")
    # Players that play with the transcript baseline instead of ego memory.
    baseline = body.get("baseline") or []
    if not isinstance(baseline, list) or not set(baseline) <= set(assignment):
        raise HTTPException(422, "baseline must list players of this match.")
    modes = {player: "baseline" if player in baseline else "ego" for player in assignment}
    try:
        scheduler = SchedulerConfig.model_validate(body.get("scheduler") or {})
    except ValidationError as exc:
        raise HTTPException(422, f"Invalid scheduler settings: {exc.errors(include_url=False)}") from exc
    return {
        "scheduler": scheduler,
        "modes": modes,
        "environment": environment,
        "assignment": assignment,
        "seed": seed,
        "impostors": impostors,
        "offline": bool(body.get("offline", False)),
        "bridge_host": "127.0.0.1",
        "bridge_port": port,
    }


def _default_model(container: Any) -> str | None:
    providers = container.runtime_context.model_providers
    return next((getattr(p, "default_model", "") for p in providers), "") or None


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(400, "Request body must be valid JSON.") from exc
    if not isinstance(body, dict):
        raise HTTPException(422, "Request body must be a JSON object.")
    return body
