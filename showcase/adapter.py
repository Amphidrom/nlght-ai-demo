"""FastAPI adapter powering the showcase catalog, runner, and portal."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from showcase.catalog import CASES, ShowcaseCase, get_case
from showcase.runs import RunRecord, RunStore
from trigger.invocation import build_http_invocation

logger = logging.getLogger(__name__)
_ASSET_DIR = Path(__file__).with_name("static")
_TERMINAL = {"succeeded", "failed", "cancelled"}


class ShowcaseAdapter:
    def __init__(self, *, run_dir: str = ".demo-runs") -> None:
        self._runs = RunStore(run_dir)

    def build_router(self) -> APIRouter:
        router = APIRouter(tags=["showcase"])

        @router.get("/demo", response_class=HTMLResponse, include_in_schema=False)
        async def portal() -> HTMLResponse:
            return HTMLResponse(_read_asset("index.html"))

        @router.get("/demo/assets/{name}", include_in_schema=False)
        async def asset(name: str) -> Response:
            if name not in {"app.js", "styles.css"}:
                raise HTTPException(status_code=404, detail="Asset not found.")
            media_type = "application/javascript" if name.endswith(".js") else "text/css"
            return Response(_read_asset(name), media_type=media_type)

        @router.get("/demo/api/cases")
        async def list_cases(request: Request) -> dict[str, Any]:
            availability = await _workflow_availability(request, CASES)
            return {
                "cases": [
                    {**case.public_dict(), "available": availability[case.workflow]}
                    for case in CASES
                ]
            }

        @router.get("/demo/api/cases/{case_id}")
        async def case_details(case_id: str, request: Request) -> dict[str, Any]:
            case = _require_case(case_id)
            availability = await _workflow_availability(request, (case,))
            return {**case.public_dict(), "available": availability[case.workflow]}

        @router.post("/demo/api/cases/{case_id}/runs", status_code=202)
        async def start_run(case_id: str, request: Request) -> JSONResponse:
            case = _require_case(case_id)
            body = await _json_body(request)
            cid = str(uuid.uuid4())
            payload = _build_payload(request, body)
            session_key = str(body.get("session_key") or cid) if case.session else None
            invocation = await build_http_invocation(
                request,
                workflow_name=case.workflow,
                payload=payload,
                correlation_id=cid,
                session_key=session_key,
                stream=True,
            )
            record = self._runs.create(
                correlation_id=cid,
                case_id=case.id,
                workflow=case.workflow,
                session_key=session_key,
            )
            task = asyncio.create_task(self._observe(request, invocation, record))
            task.add_done_callback(_consume_task_exception)
            return JSONResponse(record.public_dict(include_events=False), status_code=202)

        @router.get("/demo/api/runs/{correlation_id}")
        async def get_run(correlation_id: str) -> dict[str, Any]:
            return _require_run(self._runs, correlation_id).public_dict()

        @router.get("/demo/api/runs/{correlation_id}/events")
        async def events(correlation_id: str) -> StreamingResponse:
            record = _require_run(self._runs, correlation_id)
            return StreamingResponse(
                self._event_stream(record),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @router.post("/demo/api/runs/{correlation_id}/resume", status_code=202)
        async def resume(correlation_id: str, request: Request) -> JSONResponse:
            previous = _require_run(self._runs, correlation_id)
            if previous.status not in _TERMINAL:
                raise HTTPException(status_code=409, detail="Run is still active.")
            case = _require_case(previous.case_id)
            body = await _json_body(request)
            new_cid = str(uuid.uuid4())
            session_key = previous.session_key or previous.correlation_id
            invocation = await build_http_invocation(
                request,
                workflow_name=case.workflow,
                payload=_build_payload(request, body),
                correlation_id=new_cid,
                session_key=session_key,
                stream=True,
            )
            record = self._runs.create(
                correlation_id=new_cid,
                case_id=case.id,
                workflow=case.workflow,
                session_key=session_key,
            )
            task = asyncio.create_task(self._observe(request, invocation, record))
            task.add_done_callback(_consume_task_exception)
            return JSONResponse(record.public_dict(include_events=False), status_code=202)

        @router.get("/demo/api/health")
        async def health(request: Request) -> dict[str, Any]:
            container = request.app.state.container
            providers = [
                {"name": provider.name, "default_model": getattr(provider, "default_model", "")}
                for provider in container.runtime_context.model_providers
            ]
            return {
                "status": "ok" if container.workflow_executor else "degraded",
                "workflow_engine": container.workflow_executor is not None,
                "persistence": container.workflow_repository is not None,
                "os_runtime": (
                    getattr(container.runtime_context.os_runtime, "kind", None)
                    or type(container.runtime_context.os_runtime).__name__
                    if container.runtime_context.os_runtime else None
                ),
                "model_providers": providers,
            }

        return router

    async def _observe(self, request: Request, invocation: Any, record: RunRecord) -> None:
        executor = request.app.state.container.workflow_executor
        if executor is None:
            self._runs.set_status(record, "failed", error="Workflow engine not available.")
            return
        self._runs.set_status(record, "running")
        try:
            if hasattr(executor, "stream_signals"):
                async for signal in executor.stream_signals(invocation):
                    if signal.kind != "done":
                        self._runs.append(record, signal)
            else:
                response = await executor.execute(invocation)
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    from nlght.core.signals.signal import Signal
                    self._runs.append(record, Signal(role="assistant", content=content, kind="result"))
            self._runs.set_status(record, "succeeded")
        except asyncio.CancelledError:
            self._runs.set_status(record, "cancelled")
            raise
        except Exception as exc:
            logger.exception("showcase.run.failed | cid=%s", record.correlation_id)
            self._runs.set_status(record, "failed", error=str(exc))

    async def _event_stream(self, record: RunRecord) -> AsyncIterator[str]:
        cursor = 0
        while True:
            while cursor < len(record.events):
                event = record.events[cursor]
                cursor += 1
                yield f"event: {event['kind']}\ndata: {json.dumps(event)}\n\n"
            if record.status in _TERMINAL:
                terminal = {"status": record.status, "error": record.error}
                yield f"event: done\ndata: {json.dumps(terminal)}\n\n"
                return
            yield ": keep-alive\n\n"
            await self._runs.wait_for_change(record)


def _read_asset(name: str) -> str:
    return (_ASSET_DIR / name).read_text(encoding="utf-8")


def _require_case(case_id: str) -> ShowcaseCase:
    case = get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Unknown showcase case '{case_id}'.")
    return case


def _require_run(store: RunStore, correlation_id: str) -> RunRecord:
    record = store.get(correlation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return record


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object.")
    return body


def _build_payload(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    payload = dict(body.get("input") or body)
    payload.pop("session_key", None)
    message = payload.pop("message", None)
    if message is not None:
        payload["messages"] = [{"role": "user", "content": str(message)}]
    if not payload.get("model"):
        providers = request.app.state.container.runtime_context.model_providers
        default_model = next((getattr(p, "default_model", "") for p in providers), "")
        if default_model:
            payload["model"] = default_model
    return payload


async def _workflow_availability(
    request: Request,
    cases: tuple[ShowcaseCase, ...] | list[ShowcaseCase],
) -> dict[str, bool]:
    repository = request.app.state.container.workflow_repository
    if repository is None:
        return {case.workflow: False for case in cases}
    result: dict[str, bool] = {}
    for case in cases:
        workflow = await repository.find_by_name(case.workflow)
        result[case.workflow] = bool(workflow and workflow.enabled)
    return result


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    try:
        task.result()
    except Exception:
        logger.exception("showcase.background_task.failed")
