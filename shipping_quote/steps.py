"""Validation and explicit tool execution for a closed shipping quote."""

from __future__ import annotations

import json
from typing import ClassVar

from nlght.core.signals.signal import Signal
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext


class ShippingValidateStep(StepBase):
    TYPE: ClassVar[str] = "shipping_validate"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        shipment = ctx.trigger.payload.get("shipment", {}) if ctx.trigger else {}
        if not isinstance(shipment, dict):
            return StepResult(ctx=ctx, verdict="invalid")
        try:
            normalized = {
                "weight_kg": float(shipment["weight_kg"]),
                "zone": str(shipment["zone"]).lower(),
                "express": bool(shipment.get("express", False)),
            }
        except (KeyError, TypeError, ValueError):
            return StepResult(ctx=ctx, verdict="invalid")
        if normalized["weight_kg"] <= 0 or normalized["zone"] not in {"domestic", "eu", "international"}:
            return StepResult(ctx=ctx, verdict="invalid")
        ctx.metadata = {**ctx.metadata, "shipment": normalized}
        return StepResult(ctx=ctx, verdict="valid")


class ShippingQuoteStep(StepBase):
    TYPE: ClassVar[str] = "shipping_quote"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        if self.config.get("invalid"):
            await ctx.emitter.emit(Signal("assistant", "Shipment requires a positive weight and a valid zone.", kind="result"))
            return StepResult(ctx=ctx, verdict="done")
        tool = ctx.tools.get_or_none("shipping_quote") if ctx.tools else None
        if tool is None:
            return StepResult(ctx=ctx, verdict="failed")
        if ctx.llm is None:
            return StepResult(ctx=ctx, verdict="failed")
        await ctx.emitter.emit(Signal("assistant", "Calculating the verified shipping rate", kind="status"))
        quote = await tool.execute(**ctx.metadata["shipment"])
        system = (
            "You are a shipping assistant. Explain the verified tool quote clearly and concisely. "
            "Use only the shipment and quote supplied by the user message; do not change amounts, currency, or service facts."
        )
        user = json.dumps({"shipment": ctx.metadata["shipment"], "verified_quote": quote})
        await ctx.llm.call(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.1)
        return StepResult(ctx=ctx, verdict="done")
