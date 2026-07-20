"""Fixed-command CSV validation in the configured OS runtime."""

from __future__ import annotations

import json
from typing import ClassVar

from nlght.core.signals.signal import Signal
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext

_CHECK = """import csv,io,json,os
rows=list(csv.DictReader(io.StringIO(os.environ['CSV_INPUT'])))
columns=list(rows[0].keys()) if rows else []
missing=sum(1 for row in rows for value in row.values() if not str(value or '').strip())
print(json.dumps({'rows':len(rows),'columns':columns,'missing_values':missing}))
"""


class RuntimeCsvValidateStep(StepBase):
    TYPE: ClassVar[str] = "runtime_csv_validate"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        csv_text = ctx.trigger.payload.get("csv", "") if ctx.trigger else ""
        if not isinstance(csv_text, str) or "\n" not in csv_text or "," not in csv_text:
            return StepResult(ctx=ctx, verdict="invalid")
        ctx.metadata = {**ctx.metadata, "csv": csv_text}
        return StepResult(ctx=ctx, verdict="valid")


class RuntimeCsvCheckStep(StepBase):
    TYPE: ClassVar[str] = "runtime_csv_check"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        if self.config.get("invalid"):
            await ctx.emitter.emit(Signal("assistant", "Input must be a CSV document with a header and at least one row.", kind="result"))
            return StepResult(ctx=ctx, verdict="done")
        if ctx.os_runtime is None:
            return StepResult(ctx=ctx, verdict="failed")
        await ctx.emitter.emit(Signal("assistant", "Checking CSV data in the isolated runtime", kind="status"))
        code, stdout, stderr = await ctx.os_runtime.exec(
            ["python", "-c", _CHECK],
            env={"CSV_INPUT": ctx.metadata["csv"]},
        )
        result = {"exit_code": code, "report": stdout.strip(), "error": stderr.strip() or None}
        await ctx.emitter.emit(Signal("assistant", json.dumps(result, indent=2), kind="result"))
        return StepResult(ctx=ctx, verdict="done" if code == 0 else "failed")
