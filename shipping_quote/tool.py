"""Deterministic custom tool for the shipping quote case."""

from __future__ import annotations

import json
from typing import ClassVar

from nlght.core.model.trust import ContentOrigin, ContentTrust, SourceRef
from nlght.core.tools.action import (
    DataEgressClass,
    ScopeClass,
    SideEffectClass,
    StaticActionSemantics,
)
from nlght.core.tools.tool import ToolBase, ToolOutput, ToolParameter, ToolSignature

# The rate table is a pure in-process calculation: nothing is written, nothing
# leaves the runtime, and no state outlives the request. Declaring that is
# mandatory — a signature without action semantics is unknown, never read-only,
# and an execution-capable catalog refuses it.
_PURE_CALCULATION = StaticActionSemantics(
    side_effect=SideEffectClass.NONE,
    egress=DataEgressClass.NONE,
    scope=ScopeClass.REQUEST,
)

# The rate table is deployment code, not something a caller or a model wrote,
# so its quote is labelled as trusted content. A bare string would be
# unannotated, and unannotated counts as untrusted at every later decision.
_RATE_TABLE = ContentOrigin.of(SourceRef(
    kind="rate_table",
    resource_id="shipping_rate/local",
    source_id="shipping_quote",
    trust=ContentTrust.TRUSTED,
    name="Shipping rate table",
))


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

    async def quote(self, *, weight_kg: float, zone: str, express: bool) -> ToolOutput:
        base = {"domestic": 4.5, "eu": 8.0, "international": 15.0}[zone]
        amount = round(base + float(weight_kg) * 1.35 + (9.0 if express else 0.0), 2)
        return ToolOutput(
            content=json.dumps({"currency": "EUR", "amount": amount, "zone": zone, "express": express}),
            origin=_RATE_TABLE,
        )
