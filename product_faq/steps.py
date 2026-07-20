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
        if ctx.llm is None:
            return StepResult(ctx=ctx, verdict="failed")
        sources = list(ctx.metadata.get("faq_sources") or [])
        question = str(ctx.metadata.get("faq_question") or "")
        if not sources:
            system = (
                "You are the Northstar Desk support assistant. The retrieval step found no supporting "
                "manual article. State plainly that you cannot answer the question from the available "
                "manuals. Do not answer the underlying question and do not cite any manual. Tell the "
                "user to open Help, choose Contact Support, and send the question to a support specialist."
            )
            user = json.dumps({"question": question, "retrieved_manuals": []})
            await ctx.emitter.emit(Signal("assistant", "No grounded manual found; requesting human support", kind="status"))
            await ctx.llm.call(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.1)
            return StepResult(ctx=ctx, verdict="done")
        system = (
            "Answer the product question using only the retrieved manual records in the user message. "
            "Cite every factual sentence with the relevant manual ID in square brackets. If the records "
            "do not support the answer, say you cannot answer and direct the user to Help > Contact Support."
        )
        user = json.dumps({"question": question, "retrieved_manuals": sources})
        await ctx.emitter.emit(Signal("assistant", "Generating a grounded answer from retrieved manuals", kind="status"))
        await ctx.llm.call(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.1)
        return StepResult(ctx=ctx, verdict="done")
