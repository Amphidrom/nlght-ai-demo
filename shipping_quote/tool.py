"""Deterministic custom tool for the shipping quote case."""

from __future__ import annotations

import json
from typing import ClassVar

from nlght.core.tools.action import (
    DataEgressClass,
    ScopeClass,
    SideEffectClass,
    StaticActionSemantics,
)
from nlght.core.tools.tool import ToolBase, ToolParameter, ToolSignature

# The rate table is a pure in-process calculation: nothing is written, nothing
# leaves the runtime, and no state outlives the request. Declaring that is
# mandatory — a signature without action semantics is unknown, never read-only,
# and an execution-capable catalog refuses it.
_PURE_CALCULATION = StaticActionSemantics(
    side_effect=SideEffectClass.NONE,
    egress=DataEgressClass.NONE,
    scope=ScopeClass.REQUEST,
)


class ShippingRateTool(ToolBase):
    KIND: ClassVar[str] = "shipping_rate"
    PROVIDER: ClassVar[str] = "local"

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [ToolSignature(
            name="shipping_quote",
            description="Calculate a deterministic parcel shipping quote.",
            method_name="quote",
            parameters=[
                ToolParameter("weight_kg", "number", "Parcel weight in kilograms."),
                ToolParameter("zone", "string", "Shipping zone: domestic, eu, or international.", enum=["domestic", "eu", "international"]),
                ToolParameter("express", "boolean", "Whether express delivery is required."),
            ],
            action=_PURE_CALCULATION,
        )]

    async def quote(self, *, weight_kg: float, zone: str, express: bool) -> str:
        base = {"domestic": 4.5, "eu": 8.0, "international": 15.0}[zone]
        amount = round(base + float(weight_kg) * 1.35 + (9.0 if express else 0.0), 2)
        return json.dumps({"currency": "EUR", "amount": amount, "zone": zone, "express": express})
