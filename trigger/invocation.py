"""Shared construction of direct HTTP workflow invocations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, Request

from nlght.core.entry.context import RequestContext
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.workflow import WorkflowInvocation


async def build_http_invocation(
    request: Request,
    *,
    workflow_name: str,
    payload: dict[str, Any],
    correlation_id: str | None = None,
    session_key: str | None = None,
    stream: bool = False,
) -> WorkflowInvocation:
    cid = correlation_id or str(uuid.uuid4())
    context = RequestContext(
        correlation_id=cid,
        request_id=str(uuid.uuid4()),
        received_at=datetime.now(UTC),
        path=request.url.path,
        method=request.method,
        headers={key.lower(): value for key, value in request.headers.items()},
        query_params=dict(request.query_params),
        client_host=request.client.host if request.client else None,
    )
    return await build_invocation(
        request.app.state.container,
        workflow_name=workflow_name,
        payload=payload,
        context=context,
        session_key=session_key,
        stream=stream,
    )


async def build_invocation(
    container: Any,
    *,
    workflow_name: str,
    payload: dict[str, Any],
    context: RequestContext,
    session_key: str | None = None,
    stream: bool = False,
) -> WorkflowInvocation:
    """Resolve the active workflow version and wrap ``payload`` in a trigger.

    Used directly by in-process callers, such as the Among Us orchestrator,
    that start workflows without an HTTP request of their own.
    """

    repository = container.workflow_repository
    if repository is None or container.workflow_executor is None:
        raise HTTPException(status_code=503, detail="Workflow engine not available.")

    workflow = await repository.find_by_name(workflow_name)
    if workflow is None or not workflow.enabled:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found.")
    version = await repository.find_active_version(workflow.workflow_id)
    if version is None:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_name}' has no active version.",
        )

    trigger = Trigger(
        kind=TriggerKind.HTTP_REQUEST,
        protocol=ProtocolKind.GENERIC_JSON,
        operation=workflow_name,
        payload=payload,
        context=context,
        stream=stream,
        session_key=session_key,
    )
    return WorkflowInvocation(trigger=trigger, workflow=workflow, version=version)
