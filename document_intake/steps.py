"""Small, deterministic document-intake workflow steps."""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from nlght.core.signals.signal import Signal
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext

_CONTRACT_ID = re.compile(r"\b[A-Z]{2,8}-\d{2,8}\b")
_ORG = re.compile(r"\b([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,3})\s+(?:Ltd|GmbH|AG|Inc|LLC)\b")
_AGENCY = re.compile(r"\b((?:governmental|national|federal)\s+[a-z][a-z\s-]{2,50}?\s+agency)\b", re.IGNORECASE)


def _user_text(ctx: WorkflowStepContext) -> str:
    return next(
        (str(message.get("content", "")) for message in reversed(ctx.messages) if message.get("role") == "user"),
        "",
    )


class IntakeExtractStep(StepBase):
    TYPE: ClassVar[str] = "intake_extract"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        await ctx.emitter.emit(Signal(role="assistant", content="Extracting intake fields", kind="status"))
        text = _user_text(ctx)
        lower = text.lower()
        contract = _CONTRACT_ID.search(text)
        organization = _ORG.search(text)
        agency = _AGENCY.search(text)
        language = "de" if any(word in lower for word in ("vertrag", "verlängerung", "deutsch")) else "en"
        if "threat" in lower and ("report" in lower or "bulletin" in lower):
            document_type = "threat_bulletin"
        elif "renew" in lower or "verlänger" in lower or "contract" in lower:
            document_type = "contract_renewal"
        else:
            document_type = "general_intake"
        urgency = "high" if any(word in lower for word in ("urgent", "immediate", "critical", "dringend", "threat")) else "normal"
        organization_name = organization.group(0) if organization else agency.group(1) if agency else None
        if organization_name:
            parts = organization_name.split()
            if len(parts) > 2 and parts[0].lower() in {"today", "yesterday", "heute", "gestern"}:
                organization_name = " ".join(parts[1:])
        ctx.metadata = {
            **ctx.metadata,
            "intake": {
                "organization": organization_name,
                "contract_id": contract.group(0) if contract else None,
                "document_type": document_type,
                "language": language,
                "urgency": urgency,
                "affected_asset": "linux-based infrastructure" if "linux" in lower else None,
                "source_text": text,
            },
        }
        return StepResult(ctx=ctx, verdict="extracted")


class IntakeValidateStep(StepBase):
    TYPE: ClassVar[str] = "intake_validate"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        await ctx.emitter.emit(Signal(role="assistant", content="Validating required fields", kind="status"))
        intake = dict(ctx.metadata.get("intake") or {})
        document_type = intake.get("document_type")
        policies: list[dict[str, str]] = []
        missing: list[str] = []

        if document_type == "threat_bulletin":
            required = ("organization", "affected_asset")
            missing = [field for field in required if not intake.get(field)]
            passed = not missing and intake.get("urgency") == "high"
            policies.append({
                "policy": "SEC-01",
                "name": "Immediate infrastructure threat",
                "outcome": "passed" if passed else "failed",
                "reason": (
                    "Attributed high-urgency threat affecting named infrastructure."
                    if passed else f"Missing or insufficient fields: {', '.join(missing) or 'high urgency'}"
                ),
            })
            route = "escalated_review" if passed else "needs_review"
        elif document_type == "contract_renewal":
            required = ("organization", "contract_id")
            missing = [field for field in required if not intake.get(field)]
            policies.append({
                "policy": "CONTRACT-01",
                "name": "Renewal identifiers required",
                "outcome": "passed" if not missing else "failed",
                "reason": "Organization and contract ID are present." if not missing else f"Missing: {', '.join(missing)}",
            })
            route = "accepted" if not missing else "needs_review"
        else:
            missing = ["organization"] if not intake.get("organization") else []
            policies.append({
                "policy": "INTAKE-01",
                "name": "Attributable source required",
                "outcome": "passed" if not missing else "failed",
                "reason": "Source organization is present." if not missing else "No source organization could be identified.",
            })
            route = "accepted" if not missing else "needs_review"

        intake["missing_fields"] = missing
        intake["policy_evaluations"] = policies
        intake["routing_status"] = route
        ctx.metadata = {**ctx.metadata, "intake": intake}
        return StepResult(ctx=ctx, verdict=route)


class IntakeRespondStep(StepBase):
    TYPE: ClassVar[str] = "intake_respond"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        intake: dict[str, Any] = dict(ctx.metadata.get("intake") or {})
        expected_status = str(self.config.get("status", intake.get("routing_status", "needs_review")))
        intake["routing_status"] = expected_status
        artifact = json.dumps(intake, indent=2, ensure_ascii=False)
        await ctx.emitter.emit(Signal(role="assistant", content=artifact, kind="result"))
        return StepResult(ctx=ctx, verdict="done")
