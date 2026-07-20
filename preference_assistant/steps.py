"""Minimal Hive Mind case that stores only user-supplied preferences."""

from __future__ import annotations

import json
from typing import ClassVar

from nlght.core.hive_mind.models import AtomType, WriteIntent
from nlght.core.signals.signal import Signal
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext


class PreferenceRememberStep(StepBase):
    TYPE: ClassVar[str] = "preference_remember"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        payload = ctx.trigger.payload if ctx.trigger else {}
        action = str(payload.get("action", "remember"))
        store = ctx.store_coordinator
        if store is None:
            return StepResult(ctx=ctx, verdict="missing")
        preferences = dict(store.get_slot("demo:preferences", dict))

        if action == "apply":
            ctx.metadata = {
                **ctx.metadata,
                "preference_action": "apply",
                "preferences": preferences,
                "task": str(payload.get("task", "")),
            }
            return StepResult(ctx=ctx, verdict="apply")

        preference = payload.get("preference")
        if action != "remember" or not isinstance(preference, dict) or not preference:
            return StepResult(ctx=ctx, verdict="missing")
        key, value = next(iter(preference.items()))
        task_id = f"preference:{ctx.correlation_id}"
        store.start_task(task_id)
        store.write(WriteIntent(
            atom_type=AtomType.RESULT,
            content=json.dumps({str(key): value}),
            task_id=task_id,
            entities=[f"preference:{key}"],
            tags=["user_preference"],
            promote_immediately=True,
        ))
        store.finish_task(task_id=task_id, entities=[f"preference:{key}"], turn_nr=store.next_turn_nr())
        preferences[str(key)] = value
        store.set_slot("demo:preferences", preferences)
        ctx.metadata = {
            **ctx.metadata,
            "preference_action": "remember",
            "preferences": preferences,
        }
        return StepResult(ctx=ctx, verdict="remembered")


class PreferenceRespondStep(StepBase):
    TYPE: ClassVar[str] = "preference_respond"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        action = ctx.metadata.get("preference_action")
        preferences = dict(ctx.metadata.get("preferences") or {})
        if action == "remember":
            result = {
                "status": "remembered",
                "saved_preferences": preferences,
                "next_step": "Run the 'Step 2 · Apply' example with the same session_key.",
            }
        elif action == "apply":
            task = str(ctx.metadata.get("task") or "Explain the requested topic.")
            style = str(preferences.get("response_style", "standard"))
            if style == "concise":
                response = "Test backups regularly so recovery failures are found before an emergency."
            elif style == "detailed":
                response = (
                    "Backups only create potential recovery. Regular restore tests verify integrity, "
                    "credentials, procedures, dependencies, and the actual recovery time."
                )
            else:
                response = "Regular restore tests confirm that backups are complete and usable when recovery is required."
            result = {
                "status": "preference_applied" if preferences else "no_saved_preference",
                "task": task,
                "applied_preferences": preferences,
                "response": response,
            }
        else:
            result = {"status": "invalid_request", "expected_actions": ["remember", "apply"]}
        content = json.dumps(result, indent=2)
        await ctx.emitter.emit(Signal("assistant", content, kind="result"))
        return StepResult(ctx=ctx, verdict="done")
