"""Deterministic steps for the small incident-triage case."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from nlght.core.signals.signal import Signal
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext


class IncidentNormalizeStep(StepBase):
    TYPE: ClassVar[str] = "incident_normalize"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        await ctx.emitter.emit(Signal("assistant", "Normalizing the incident signal", kind="status"))
        raw = ctx.trigger.payload.get("alert", {}) if ctx.trigger else {}
        alert = raw if isinstance(raw, dict) else {}
        try:
            error_rate = max(0.0, float(alert.get("error_rate", 0)))
            p95_ms = max(0, int(alert.get("p95_ms", 0)))
        except (TypeError, ValueError):
            return StepResult(ctx=ctx, verdict="invalid")
        ctx.metadata = {
            **ctx.metadata,
            "incident": {
                "service": str(alert.get("service") or "unknown"),
                "error_rate": error_rate,
                "p95_ms": p95_ms,
            },
        }
        return StepResult(ctx=ctx, verdict="normalized")


class IncidentClassifyStep(StepBase):
    TYPE: ClassVar[str] = "incident_classify"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        await ctx.emitter.emit(Signal("assistant", "Applying severity thresholds", kind="status"))
        incident = dict(ctx.metadata.get("incident") or {})
        error_rate = float(incident.get("error_rate", 0))
        p95_ms = int(incident.get("p95_ms", 0))
        if error_rate >= 0.2 or p95_ms >= 3000:
            severity = "critical"
        elif error_rate >= 0.1 or p95_ms >= 1500:
            severity = "high"
        else:
            severity = "moderate"
        incident["severity"] = severity
        ctx.metadata = {**ctx.metadata, "incident": incident}
        return StepResult(ctx=ctx, verdict="urgent" if severity in {"critical", "high"} else "routine")


class IncidentRespondStep(StepBase):
    TYPE: ClassVar[str] = "incident_respond"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        incident: dict[str, Any] = dict(ctx.metadata.get("incident") or {})
        route = str(self.config.get("route", "manual_review"))
        actions = (
            ["Page the service owner", "Check the latest deployment", "Protect downstream dependencies"]
            if route == "urgent_response"
            else ["Create a triage ticket", "Inspect recent service metrics"]
        )
        result = {**incident, "route": route, "recommended_actions": actions}
        await ctx.emitter.emit(Signal("assistant", json.dumps(result, indent=2), kind="result"))
        return StepResult(ctx=ctx, verdict="done")
