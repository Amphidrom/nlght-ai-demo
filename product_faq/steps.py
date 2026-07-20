"""Small grounded FAQ case with explicit source identifiers."""

from __future__ import annotations

import json
import re
from typing import ClassVar

from nlght.core.signals.signal import Signal
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext

_ARTICLES = (
    {"id": "MANUAL-101", "title": "Offline mode", "topics": ("offline", "draft", "drafts", "sync", "synchronize", "connectivity"), "text": "Northstar Desk keeps drafts locally while offline and synchronizes them after connectivity returns."},
    {"id": "MANUAL-204", "title": "Data retention", "topics": ("deleted", "deletion", "recover", "recoverable", "recovery", "retention", "restore", "removal"), "text": "Deleted Northstar Desk projects remain recoverable for 30 days before permanent removal."},
    {"id": "MANUAL-310", "title": "Team roles", "topics": ("role", "roles", "owner", "editor", "commenter", "viewer", "permission", "permissions"), "text": "Northstar Desk supports Owner, Editor, Commenter, and Viewer roles."},
)


class ProductRetrieveStep(StepBase):
    TYPE: ClassVar[str] = "product_faq_retrieve"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        await ctx.emitter.emit(Signal("assistant", "Searching the product manual", kind="status"))
        question = next((str(m.get("content", "")) for m in reversed(ctx.messages) if m.get("role") == "user"), "")
        terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        matches = [article for article in _ARTICLES if terms & set(article["topics"])]
        ctx.metadata = {**ctx.metadata, "faq_question": question, "faq_sources": matches}
        return StepResult(ctx=ctx, verdict="found" if matches else "not_found")


class ProductAnswerStep(StepBase):
    TYPE: ClassVar[str] = "product_faq_answer"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        sources = list(ctx.metadata.get("faq_sources") or [])
        if not sources:
            result = {
                "status": "human_review_required",
                "answer": "I cannot answer this question from the available Northstar Desk manuals.",
                "reason": "No manual article supports an answer to this question.",
                "support": {
                    "channel": "Northstar Desk application",
                    "instructions": "Open Help, select Contact Support, and send this question to a support specialist.",
                },
                "question": str(ctx.metadata.get("faq_question") or ""),
                "sources": [],
            }
            await ctx.emitter.emit(Signal("assistant", json.dumps(result, indent=2), kind="result"))
            return StepResult(ctx=ctx, verdict="done")
        citations = ", ".join(f"[{source['id']}]" for source in sources)
        facts = " ".join(source["text"] for source in sources)
        await ctx.emitter.emit(Signal("assistant", f"{facts}\n\nSources: {citations}", kind="result"))
        return StepResult(ctx=ctx, verdict="done")
