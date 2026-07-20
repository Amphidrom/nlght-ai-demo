from __future__ import annotations

import unittest
import tomllib
from types import SimpleNamespace
from pathlib import Path

from document_intake.steps import IntakeExtractStep, IntakeRespondStep, IntakeValidateStep
from incident_triage.steps import IncidentClassifyStep, IncidentNormalizeStep, IncidentRespondStep
from preference_assistant.steps import PreferenceRememberStep, PreferenceRespondStep
from product_faq.steps import ProductAnswerStep, ProductRetrieveStep
from shipping_quote.steps import ShippingQuoteStep
from shipping_quote.tool import ShippingRateTool
from runtime_data_check.steps import RuntimeCsvCheckStep, RuntimeCsvValidateStep
from nlght.core.signals.signal import Signal
from nlght.core.workflow.step import WorkflowStepContext
from showcase.adapter import ShowcaseAdapter
from showcase.catalog import CASES
from showcase.runs import RunStore


class Emitter:
    def __init__(self) -> None:
        self.signals: list[Signal] = []

    async def emit(self, signal: Signal) -> None:
        self.signals.append(signal)


class RecordingLlm:
    def __init__(self, emitter: Emitter, response: str = "model response") -> None:
        self.emitter = emitter
        self.response = response
        self.calls: list[dict] = []

    async def call(self, messages, *, temperature=None) -> None:
        self.calls.append({"messages": messages, "temperature": temperature})
        await self.emitter.emit(Signal("assistant", self.response, kind="result"))


def with_llm(ctx: WorkflowStepContext, response: str = "model response") -> RecordingLlm:
    llm = RecordingLlm(ctx.emitter, response)  # type: ignore[arg-type]
    ctx.llm = llm  # type: ignore[assignment]
    return llm


class FakeStoreCoordinator:
    def __init__(self) -> None:
        self.slots: dict[str, object] = {}
        self.writes: list[object] = []

    def get_slot(self, key: str, expected_type=None):
        return self.slots.get(key, expected_type() if expected_type else None)

    def set_slot(self, key: str, value: object) -> None:
        self.slots[key] = value

    def start_task(self, task_id: str) -> None:
        self.task_id = task_id

    def write(self, intent: object) -> None:
        self.writes.append(intent)

    def finish_task(self, **kwargs) -> None:
        self.finished = kwargs

    def next_turn_nr(self) -> int:
        return 1


def context(message: str, *, payload: dict | None = None) -> WorkflowStepContext:
    return WorkflowStepContext(
        correlation_id="test-cid",
        trigger=SimpleNamespace(payload=payload or {}),  # type: ignore[arg-type]
        model="",
        messages=[{"role": "user", "content": message}],
        stream=False,
        emitter=Emitter(),
    )


class CatalogTests(unittest.TestCase):
    def test_catalog_has_six_unique_non_security_cases(self) -> None:
        self.assertEqual(6, len(CASES))
        self.assertEqual(6, len({case.id for case in CASES}))
        self.assertEqual(
            {
                "product-faq",
                "shipping-quote",
                "incident-triage",
                "document-intake",
                "preference-assistant",
                "runtime-data-check",
            },
            {case.id for case in CASES},
        )

    def test_adapter_returns_router_with_public_routes(self) -> None:
        paths = {route.path for route in ShowcaseAdapter().build_router().routes}
        self.assertIn("/demo", paths)
        self.assertIn("/demo/api/cases", paths)
        self.assertIn("/demo/api/runs/{correlation_id}/events", paths)

    def test_seed_defines_only_the_registered_shipping_resource(self) -> None:
        seed_text = "\n".join(path.read_text(encoding="utf-8") for path in Path("seed").glob("*.sql"))
        self.assertEqual(1, seed_text.count("INSERT INTO resources"))
        self.assertIn("'shipping_rate'", seed_text)
        self.assertEqual("shipping_rate", ShippingRateTool.KIND)

    def test_every_llm_terminal_step_selects_its_provider(self) -> None:
        seed_text = "\n".join(path.read_text(encoding="utf-8") for path in Path("seed").glob("*.sql"))
        self.assertEqual(10, seed_text.count('"model_provider":"ollama"'))

    def test_portal_uses_static_brand_styles_without_motion(self) -> None:
        styles = Path("showcase/static/styles.css").read_text(encoding="utf-8")
        markup = Path("showcase/static/index.html").read_text(encoding="utf-8")
        self.assertIn("--paper: #010c0e", styles)
        self.assertIn("--accent: #00c8a0", styles)
        self.assertIn("--lime: #00f0d0", styles)
        for motion_rule in ("transition:", "animation:", "@keyframes", "scroll-behavior:"):
            self.assertNotIn(motion_rule, styles)
        self.assertIn("© 2026 Amphidrom GmbH", markup)

    def test_legacy_demo_sources_are_absent(self) -> None:
        legacy_paths = (
            "github_pr", "memory", "prompt", "rag", "research", "runtime",
            "tool", "translate", "trigger/adapter.py", "file_outline.py",
            "status_updates.py", "requests.rest", "requests.sh",
        )
        self.assertFalse([path for path in legacy_paths if Path(path).exists()])

    def test_readme_documents_ollama_models_and_startup(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("ollama serve", readme)
        self.assertIn("ollama pull qwen3:30b-a3b-q4_K_M", readme)
        self.assertIn("ollama pull qwen3-coder:30b", readme)
        self.assertIn("http://localhost:11434", readme)

    def test_release_package_includes_cli_and_portal_assets(self) -> None:
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual("main:run", project["project"]["scripts"]["nlght-ai-demo"])
        self.assertEqual(["main"], project["tool"]["setuptools"]["py-modules"])
        self.assertEqual(
            ["static/*.html", "static/*.css", "static/*.js"],
            project["tool"]["setuptools"]["package-data"]["showcase"],
        )

    def test_release_workflow_is_tag_triggered_and_install_verified(self) -> None:
        workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
        self.assertIn('"v[0-9]*"', workflow)
        self.assertIn("python -m build", workflow)
        self.assertIn("pip install dist/*.whl", workflow)
        self.assertIn("gh release create", workflow)
        self.assertIn("contents: write", workflow)


class RunStoreTests(unittest.TestCase):
    def test_run_collects_results_and_terminal_status(self) -> None:
        store = RunStore("Z:/path-that-is-not-required-for-live-runs")
        record = store.create(correlation_id="run-1", case_id="product-faq", workflow="product_faq")
        store.append(record, Signal(role="assistant", content="answer", kind="result"))
        store.set_status(record, "succeeded")
        self.assertEqual("answer", record.result)
        self.assertEqual("succeeded", record.status)
        self.assertEqual("result", record.events[0]["kind"])


class DocumentIntakeTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_intake_routes_to_accepted(self) -> None:
        ctx = context("Today Acme Ltd submitted a renewal for contract AC-1042.")
        extracted = await IntakeExtractStep(config={}).run(ctx)
        validated = await IntakeValidateStep(config={}).run(extracted.ctx)
        self.assertEqual("accepted", validated.verdict)
        llm = with_llm(validated.ctx, "Contract AC-1042 passed CONTRACT-01 and was accepted.")
        response = await IntakeRespondStep(config={"status": "accepted"}).run(validated.ctx)
        self.assertEqual("done", response.verdict)
        self.assertIn("AC-1042", llm.calls[0]["messages"][1]["content"])
        self.assertIn("CONTRACT-01", llm.calls[0]["messages"][1]["content"])

    async def test_missing_fields_routes_to_manual_review(self) -> None:
        ctx = context("A renewal request arrived today.")
        extracted = await IntakeExtractStep(config={}).run(ctx)
        validated = await IntakeValidateStep(config={}).run(extracted.ctx)
        self.assertEqual("needs_review", validated.verdict)
        self.assertEqual(["organization", "contract_id"], validated.ctx.metadata["intake"]["missing_fields"])

    async def test_immediate_linux_threat_passes_policy_and_escalates(self) -> None:
        message = (
            "Today the governmental cybersecurity agency released a report on an immediate "
            "threat against all linux based infrastructure."
        )
        extracted = await IntakeExtractStep(config={}).run(context(message))
        validated = await IntakeValidateStep(config={}).run(extracted.ctx)
        intake = validated.ctx.metadata["intake"]
        self.assertEqual("escalated_review", validated.verdict)
        self.assertEqual("threat_bulletin", intake["document_type"])
        self.assertEqual("high", intake["urgency"])
        self.assertEqual("governmental cybersecurity agency", intake["organization"])
        self.assertEqual("linux-based infrastructure", intake["affected_asset"])
        self.assertEqual("SEC-01", intake["policy_evaluations"][0]["policy"])
        self.assertEqual("passed", intake["policy_evaluations"][0]["outcome"])


class PreferenceAssistantTests(unittest.IsolatedAsyncioTestCase):
    async def test_remembered_preference_changes_later_response(self) -> None:
        store = FakeStoreCoordinator()
        remember_ctx = context("Remember my preference", payload={
            "action": "remember",
            "preference": {"response_style": "concise"},
        })
        remember_ctx.store_coordinator = store  # type: ignore[assignment]
        remembered = await PreferenceRememberStep(config={}).run(remember_ctx)
        self.assertEqual("remembered", remembered.verdict)
        self.assertEqual({"response_style": "concise"}, store.slots["demo:preferences"])

        apply_ctx = context("Apply my preference", payload={
            "action": "apply",
            "task": "Explain why restore tests matter.",
        })
        apply_ctx.store_coordinator = store  # type: ignore[assignment]
        applied = await PreferenceRememberStep(config={}).run(apply_ctx)
        llm = with_llm(applied.ctx, "Test backups regularly.")
        response = await PreferenceRespondStep(config={}).run(applied.ctx)
        result = response.ctx.emitter.signals[-1].content
        self.assertIn("Test backups regularly", result)
        prompt = llm.calls[0]["messages"][1]["content"]
        self.assertIn('"response_style": "concise"', prompt)
        self.assertIn("Explain why restore tests matter", prompt)


class ProductFaqTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_contains_explicit_manual_source(self) -> None:
        ctx = context("How long can I recover a deleted project?")
        retrieved = await ProductRetrieveStep(config={}).run(ctx)
        self.assertEqual("found", retrieved.verdict)
        llm = with_llm(retrieved.ctx, "Projects remain recoverable for 30 days. [MANUAL-204]")
        answered = await ProductAnswerStep(config={}).run(retrieved.ctx)
        self.assertIn("[MANUAL-204]", answered.ctx.emitter.signals[-1].content)
        self.assertIn("MANUAL-204", llm.calls[0]["messages"][1]["content"])

    async def test_unsupported_question_is_not_answered(self) -> None:
        ctx = context("Does it support satellite telemetry?")
        retrieved = await ProductRetrieveStep(config={}).run(ctx)
        self.assertEqual("not_found", retrieved.verdict)
        llm = with_llm(
            retrieved.ctx,
            "I cannot answer this question from the available manuals. Open Help > Contact Support.",
        )
        answered = await ProductAnswerStep(config={}).run(retrieved.ctx)
        result = answered.ctx.emitter.signals[-1].content
        self.assertIn("I cannot answer this question", result)
        self.assertIn("Contact Support", result)
        self.assertNotIn("MANUAL-", result)
        self.assertIn('"retrieved_manuals": []', llm.calls[0]["messages"][1]["content"])

    async def test_product_name_and_support_are_not_retrieval_evidence(self) -> None:
        ctx = context("Does Northstar Desk support satellite telemetry?")
        retrieved = await ProductRetrieveStep(config={}).run(ctx)
        self.assertEqual([], retrieved.ctx.metadata["faq_sources"])

    async def test_combined_question_retrieves_only_both_relevant_articles(self) -> None:
        ctx = context("Can I work offline and which team roles are available?")
        retrieved = await ProductRetrieveStep(config={}).run(ctx)
        self.assertEqual("found", retrieved.verdict)
        self.assertEqual(
            ["MANUAL-101", "MANUAL-310"],
            [source["id"] for source in retrieved.ctx.metadata["faq_sources"]],
        )


class ShippingToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_quote_is_deterministic(self) -> None:
        tool = ShippingRateTool(name="shipping-rates", config={})
        result = await tool.quote(weight_kg=3.5, zone="eu", express=True)
        self.assertIn('"amount": 21.73', result)

    async def test_quote_step_passes_tool_result_to_model(self) -> None:
        class QuoteContract:
            async def execute(self, **shipment):
                return '{"amount": 21.73, "currency": "EUR"}'

        ctx = context("", payload={"shipment": {"weight_kg": 3.5, "zone": "eu", "express": True}})
        ctx.metadata = {"shipment": {"weight_kg": 3.5, "zone": "eu", "express": True}}
        ctx.tools = SimpleNamespace(get_or_none=lambda name: QuoteContract())  # type: ignore[assignment]
        llm = with_llm(ctx, "The verified rate is EUR 21.73.")
        result = await ShippingQuoteStep(config={}).run(ctx)
        self.assertEqual("done", result.verdict)
        self.assertIn("21.73", llm.calls[0]["messages"][1]["content"])


class RuntimeDataCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_csv_is_passed_to_fixed_process_environment(self) -> None:
        class Runtime:
            async def exec(self, command, *, cwd=None, env=None):
                self.command = command
                self.cwd = cwd
                self.env = env
                return 0, '{"rows": 2, "columns": ["name", "email"], "missing_values": 1}\n', ""

        ctx = context("", payload={"csv": "name,email\nAda,ada@example.test\nGrace,"})
        runtime = Runtime()
        ctx.os_runtime = runtime
        validated = await RuntimeCsvValidateStep(config={}).run(ctx)
        llm = with_llm(validated.ctx, "The CSV has 2 rows and 1 missing value.")
        checked = await RuntimeCsvCheckStep(config={}).run(validated.ctx)
        self.assertEqual("done", checked.verdict)
        self.assertEqual(ctx.metadata["csv"], runtime.env["CSV_INPUT"])
        self.assertEqual(["python", "-c"], runtime.command[:2])
        self.assertEqual("/", runtime.cwd)
        self.assertIn('"missing_values": 1', llm.calls[0]["messages"][1]["content"])

    async def test_docker_runtime_config_does_not_require_a_workspace_mount(self) -> None:
        config = Path("docker/platform.yaml").read_text(encoding="utf-8")
        self.assertIn('base_image: "python:3.13-slim"', config)
        self.assertNotIn("workspace_path:", config)


class IncidentTriageTests(unittest.IsolatedAsyncioTestCase):
    async def test_high_error_rate_routes_to_urgent_response(self) -> None:
        ctx = context("Triage alert", payload={
            "alert": {"service": "checkout", "error_rate": 0.18, "p95_ms": 2400}
        })
        normalized = await IncidentNormalizeStep(config={}).run(ctx)
        classified = await IncidentClassifyStep(config={}).run(normalized.ctx)
        self.assertEqual("urgent", classified.verdict)
        llm = with_llm(classified.ctx, "High severity. Page the checkout owner.")
        response = await IncidentRespondStep(config={"route": "urgent_response"}).run(classified.ctx)
        self.assertEqual("done", response.verdict)
        self.assertIn('"severity": "high"', llm.calls[0]["messages"][1]["content"])
        self.assertIn('"selected_route": "urgent_response"', llm.calls[0]["messages"][1]["content"])

    async def test_invalid_metrics_take_manual_review_path(self) -> None:
        ctx = context("Triage alert", payload={"alert": {"error_rate": "not-a-number"}})
        normalized = await IncidentNormalizeStep(config={}).run(ctx)
        self.assertEqual("invalid", normalized.verdict)


if __name__ == "__main__":
    unittest.main()
