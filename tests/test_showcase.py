from __future__ import annotations

import json
import re
import tempfile
import unittest
import tomllib
from types import SimpleNamespace
from pathlib import Path

from document_intake.steps import IntakeExtractStep, IntakeRespondStep, IntakeValidateStep
from incident_triage.steps import IncidentClassifyStep, IncidentNormalizeStep, IncidentRespondStep
from knowledge_base.steps import (
    KnowledgeAnswerStep,
    KnowledgeIngestReportStep,
    KnowledgeNothingFoundStep,
    KnowledgeRouteStep,
)
from preference_assistant.steps import PreferenceRememberStep, PreferenceRespondStep
from product_faq.steps import ProductAnswerStep, ProductRetrieveStep
from shipping_quote.steps import ShippingQuoteStep
from shipping_quote.tool import ShippingRateTool
from runtime_data_check.steps import RuntimeCsvCheckStep, RuntimeCsvValidateStep
from nlght.core.model.messages import (
    TrustedInstructionMessage,
    UntrustedContextMessage,
    UserMessage,
)
from nlght.core.model.trust import ContentOrigin, ContentTrust, SourceRef
from nlght.core.signals.signal import Signal
from nlght.core.tools.tool import ToolOutput
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

    async def call(self, messages) -> None:
        # Sampling is provider configuration since nlght-ai 0.1.4: a step that
        # still passes temperature fails here exactly as it would in the runtime.
        self.calls.append({"messages": messages})
        await self.emitter.emit(Signal("assistant", self.response, kind="result"))


def instructions_of(call: dict) -> str:
    """The single deployment-authored message — the only trusted channel."""
    trusted = [m for m in call["messages"] if isinstance(m, TrustedInstructionMessage)]
    assert len(trusted) == 1, f"expected exactly one trusted message, got {len(trusted)}"
    return trusted[0].content


def evidence_of(call: dict) -> str:
    """Every untrusted context record the step offered, as one blob."""
    return "\n".join(
        record.content
        for message in call["messages"]
        if isinstance(message, UntrustedContextMessage)
        for record in message.records
    )


def request_of(call: dict) -> str:
    return "\n".join(
        str(message.content) for message in call["messages"] if isinstance(message, UserMessage)
    )


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
        messages=[UserMessage(message)],
        stream=False,
        emitter=Emitter(),
    )


class CatalogTests(unittest.TestCase):
    def test_catalog_has_seven_unique_non_security_cases(self) -> None:
        self.assertEqual(7, len(CASES))
        self.assertEqual(7, len({case.id for case in CASES}))
        self.assertEqual(
            {
                "product-faq",
                "shipping-quote",
                "incident-triage",
                "document-intake",
                "preference-assistant",
                "runtime-data-check",
                "knowledge-base",
            },
            {case.id for case in CASES},
        )

    def test_only_the_knowledge_case_needs_the_optional_stack(self) -> None:
        needing = {case.id for case in CASES if case.optional_integrations}
        self.assertEqual({"knowledge-base"}, needing)
        knowledge = next(case for case in CASES if case.id == "knowledge-base")
        self.assertTrue(knowledge.advanced)
        self.assertEqual(("knowledge database", "OpenSearch"), knowledge.optional_integrations)

    def test_default_seed_leaves_the_knowledge_case_unseeded(self) -> None:
        # An unseeded workflow is what makes the portal disable the case, so the
        # optional stack is never a failed click.
        entrypoint = Path("seed.sql").read_text(encoding="utf-8")
        self.assertNotIn("\\i demo/seed/07_knowledge_base.sql", entrypoint)
        self.assertIn("07_knowledge_base.sql", entrypoint)
        self.assertTrue(Path("seed/07_knowledge_base.sql").is_file())

    def test_portal_explains_why_an_optional_case_is_unavailable(self) -> None:
        app_js = Path("showcase/static/app.js").read_text(encoding="utf-8")
        self.assertIn("optional_integrations", app_js)
        self.assertIn("Needs the optional stack", app_js)
        self.assertIn(".requires", Path("showcase/static/styles.css").read_text(encoding="utf-8"))

    def test_adapter_returns_router_with_public_routes(self) -> None:
        paths = {route.path for route in ShowcaseAdapter().build_router().routes}
        self.assertIn("/demo", paths)
        self.assertIn("/demo/api/cases", paths)
        self.assertIn("/demo/api/runs/{correlation_id}/events", paths)

    def test_every_seeded_resource_activates_a_registered_tool_kind(self) -> None:
        import main  # noqa: F401  — registers the demo's own tools
        from nlght.adapters.outbound.tools.registry import tool_registry

        seed_text = "\n".join(path.read_text(encoding="utf-8") for path in Path("seed").glob("*.sql"))
        seeded = set(re.findall(r"'[^']*',\s*'([a-z_]+)',\s*'(?:local|postgres\+opensearch|qdrant\+opensearch)'", seed_text))
        registered = {kind for kind, _ in getattr(tool_registry, "_registry")}
        self.assertIn("shipping_rate", seeded)
        self.assertEqual("shipping_rate", ShippingRateTool.KIND)
        self.assertEqual(set(), seeded - registered, "seed activates a resource kind no tool registers")

    def test_only_the_knowledge_seed_activates_a_store(self) -> None:
        base = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(Path("seed").glob("*.sql"))
            if path.name != "07_knowledge_base.sql"
        )
        self.assertNotIn("knowledge_store", base)
        self.assertNotIn("knowledge_index_writer", base)
        knowledge = Path("seed/07_knowledge_base.sql").read_text(encoding="utf-8")
        self.assertIn("'knowledge_store'", knowledge)
        self.assertIn("'knowledge_index_writer'", knowledge)

    def test_every_llm_terminal_step_selects_its_provider(self) -> None:
        seed_text = "\n".join(path.read_text(encoding="utf-8") for path in Path("seed").glob("*.sql"))
        # Six cases contribute ten terminal LLM steps; the knowledge case adds
        # its answer step. Its extraction step reads through the provider held
        # at temperature 0.
        # The two Among Us workflows (ego and baseline) add one decide step each.
        self.assertEqual(13, seed_text.count('"model_provider":"ollama"'))
        self.assertEqual(1, seed_text.count('"model_provider":"ollama-extract"'))

    def test_sampling_is_provider_configuration(self) -> None:
        seed_text = "\n".join(path.read_text(encoding="utf-8") for path in Path("seed").glob("*.sql"))
        self.assertNotIn('"temperature"', seed_text)
        for path in Path(".").glob("*/steps.py"):
            self.assertNotIn("temperature", path.read_text(encoding="utf-8"), path)
        config = Path("docker/platform.yaml").read_text(encoding="utf-8")
        self.assertRegex(config, r'name: "ollama"\n(?:.*\n){4}\s+temperature: 0\.1')
        self.assertRegex(config, r'name: "ollama-extract"\n(?:.*\n){4}\s+temperature: 0\n')

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

    def test_readme_documents_the_optional_knowledge_stack(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("docker-compose.knowledge.yml", readme)
        self.assertIn("migrate-knowledge", readme)
        self.assertIn("uv sync --extra knowledge", readme)
        self.assertIn("seed/07_knowledge_base.sql", readme)
        # The relaxed review policy must never read as the platform default.
        self.assertIn("demo choice, not a production default", readme.lower())
        self.assertIn("no vector store and no embedding provider", readme)

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
        # A directory the store never has to create: the suite must not write
        # into the repository, and a hardcoded Windows path becomes a real
        # "Z:" directory here on macOS and Linux.
        store = RunStore(str(Path(tempfile.mkdtemp()) / "runs"))
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
        self.assertIn("AC-1042", evidence_of(llm.calls[0]))
        self.assertIn("CONTRACT-01", evidence_of(llm.calls[0]))
        self.assertNotIn("AC-1042", instructions_of(llm.calls[0]))

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
        self.assertIn('"response_style": "concise"', evidence_of(llm.calls[0]))
        self.assertNotIn("concise", instructions_of(llm.calls[0]))
        self.assertIn("Explain why restore tests matter", request_of(llm.calls[0]))


class ProductFaqTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_contains_explicit_manual_source(self) -> None:
        ctx = context("How long can I recover a deleted project?")
        retrieved = await ProductRetrieveStep(config={}).run(ctx)
        self.assertEqual("found", retrieved.verdict)
        llm = with_llm(retrieved.ctx, "Projects remain recoverable for 30 days. [MANUAL-204]")
        answered = await ProductAnswerStep(config={}).run(retrieved.ctx)
        self.assertIn("[MANUAL-204]", answered.ctx.emitter.signals[-1].content)
        self.assertIn("MANUAL-204", evidence_of(llm.calls[0]))
        self.assertIn("How long can I recover a deleted project?", request_of(llm.calls[0]))

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
        self.assertEqual("", evidence_of(llm.calls[0]))
        self.assertNotIn("MANUAL-", instructions_of(llm.calls[0]))

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
        self.assertIn('"amount": 21.73', result.content)

    async def test_quote_is_labelled_as_trusted_rate_table_content(self) -> None:
        tool = ShippingRateTool(name="shipping-rates", config={})
        result = await tool.quote(weight_kg=1.0, zone="domestic", express=False)
        self.assertIs(ContentTrust.TRUSTED, result.origin.trust)
        self.assertEqual(
            [("rate_table", "shipping_rate/local", "shipping_quote")],
            [ref.identity for ref in result.origin.sources],
        )

    async def test_signature_declares_action_semantics(self) -> None:
        from nlght.adapters.outbound.tools.registry import ToolLoader
        from nlght.core.tools.action import (
            DataEgressClass,
            ScopeClass,
            SideEffectClass,
        )

        signature = ShippingRateTool.signatures()[0]
        self.assertIsNotNone(signature.action)
        self.assertEqual(SideEffectClass.NONE, signature.action.side_effect)
        self.assertEqual(DataEgressClass.NONE, signature.action.egress)
        self.assertEqual(ScopeClass.REQUEST, signature.action.scope)
        # An execution-capable catalog refuses a signature with unknown
        # semantics, so registration is the real assertion here.
        ToolLoader().register(ShippingRateTool)

    async def test_quote_step_passes_tool_result_to_model(self) -> None:
        class FakeCatalog:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def execute_with_origin(self, tc, *, context_origin=None):
                self.calls.append(tc)
                self.context_origin = context_origin
                return ToolOutput('{"amount": 21.73, "currency": "EUR"}', origin=rate_table)

        rate_table = ContentOrigin.of(
            SourceRef("rate_table", "shipping_rate/local", "shipping_quote", ContentTrust.TRUSTED),
        )

        shipment = {"weight_kg": 3.5, "zone": "eu", "express": True}
        ctx = context("", payload={"shipment": shipment})
        ctx.metadata = {"shipment": shipment}
        catalog = FakeCatalog()
        ctx.tools = catalog  # type: ignore[assignment]
        llm = with_llm(ctx, "The verified rate is EUR 21.73.")
        result = await ShippingQuoteStep(config={}).run(ctx)
        self.assertEqual("done", result.verdict)
        self.assertEqual([{"name": "shipping_quote", "input": shipment}], catalog.calls)
        self.assertIn("21.73", evidence_of(llm.calls[0]))
        # The decision saw the caller's shipment as untrusted content, and the
        # model receives the quote with the origin the tool reported.
        self.assertTrue(catalog.context_origin.untrusted_sources)
        records = [
            record
            for message in llm.calls[0]["messages"]
            if isinstance(message, UntrustedContextMessage)
            for record in message.records
        ]
        self.assertEqual([rate_table], [record.provenance.origin for record in records])


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
        self.assertIn('"missing_values": 1', evidence_of(llm.calls[0]))

    async def test_docker_runtime_config_does_not_require_a_workspace_mount(self) -> None:
        config = Path("docker/platform.yaml").read_text(encoding="utf-8")
        self.assertIn('base_image: "python:3.13-slim"', config)
        self.assertNotIn("workspace_path:", config)


class KnowledgeBaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_and_ask_are_separate_entry_points(self) -> None:
        ingest = await KnowledgeRouteStep(config={}).run(context("", payload={"action": "ingest"}))
        self.assertEqual("ingest", ingest.verdict)

        ask = await KnowledgeRouteStep(config={}).run(
            context("Which team roles exist?", payload={"action": "ask"})
        )
        self.assertEqual("ask", ask.verdict)

    async def test_a_request_without_an_action_ends_with_guidance(self) -> None:
        result = await KnowledgeRouteStep(config={}).run(context("", payload={}))
        self.assertEqual("done", result.verdict)
        self.assertIn("ingest", result.ctx.emitter.signals[-1].content)

    async def test_asking_without_a_question_does_not_reach_the_stores(self) -> None:
        result = await KnowledgeRouteStep(config={}).run(context("", payload={"action": "ask"}))
        self.assertEqual("done", result.verdict)
        self.assertIn("Send a question", result.ctx.emitter.signals[-1].content)

    async def test_report_counts_what_the_run_established(self) -> None:
        ctx = context("", payload={"action": "ingest"})
        ctx.metadata = {
            "ingestion.snapshot": SimpleNamespace(documents=("a", "b", "c")),
            "knowledge.units": ("u1", "u2"),
            "knowledge.extracted": ("c1", "c2", "c3"),
            "knowledge.classified": (("c1", None), ("c2", None), ("c3", "held back")),
            "knowledge.persisted": ("c1", "c2"),
            "knowledge.retired": (),
        }
        result = await KnowledgeIngestReportStep(config={}).run(ctx)
        report = json.loads(result.ctx.emitter.signals[-1].content)
        self.assertEqual(3, report["documents_acquired"])
        self.assertEqual(3, report["claims_extracted"])
        self.assertEqual(1, report["claims_quarantined"])
        self.assertEqual(2, report["claims_persisted"])

    async def test_nothing_found_names_the_two_causes_it_cannot_separate(self) -> None:
        ctx = context("Does it support satellite telemetry?", payload={"action": "ask"})
        result = await KnowledgeNothingFoundStep(config={}).run(ctx)
        payload = json.loads(result.ctx.emitter.signals[-1].content)
        self.assertIn("No reviewed claim", payload["answer"])
        self.assertIn("ingest", payload["next_step"])

    async def test_answer_step_calls_the_model_with_the_composed_messages(self) -> None:
        ctx = context("Which team roles exist?", payload={"action": "ask"})
        llm = with_llm(ctx, "Owner, Editor, Commenter, Viewer.")
        result = await KnowledgeAnswerStep(config={}).run(ctx)
        self.assertEqual("done", result.verdict)
        # prompt.compose owns the message assembly; this step must not rebuild it.
        self.assertEqual(list(ctx.messages), list(llm.calls[0]["messages"]))


class KnowledgeSeedTests(unittest.TestCase):
    def _seed(self) -> str:
        return Path("seed/07_knowledge_base.sql").read_text(encoding="utf-8")

    def test_every_seeded_step_type_is_registered(self) -> None:
        import main  # noqa: F401  — registers the demo's own steps
        from nlght.adapters.outbound.workflow.registry import step_registry

        registered = set(getattr(step_registry, "_registry"))
        seeded = set(re.findall(r"',\s*'([a-z_]+(?:\.[a-z_]+)?)',\n?\s*'\{", self._seed()))
        self.assertTrue(seeded, "no step types parsed out of the knowledge seed")
        self.assertEqual(set(), seeded - registered, "seeded step types missing from the registry")

    def test_every_transition_target_is_a_step_in_this_workflow(self) -> None:
        seed = self._seed()
        defined = set(re.findall(r"\('(b7100000-0000-0000-0000-0000000000\d\d)',\s*'b7100000", seed))
        targets = set(re.findall(r'":"(b7100000-[0-9a-f-]+)"', seed))
        self.assertTrue(targets)
        self.assertEqual(set(), targets - defined, "transition points at an undefined step")

    def test_retrieval_branch_needs_no_vector_store(self) -> None:
        seed = self._seed()
        self.assertIn('"sources":["knowledge"]', seed)
        self.assertNotIn("qdrant", seed)
        self.assertNotIn("data_store", seed)

    def test_store_trusts_only_the_ingested_handbook(self) -> None:
        seed = self._seed()
        self.assertIn('"source_id":"northstar-handbook"', seed)
        self.assertIn('"trusted_sources":["northstar-handbook"]', seed)
        self.assertEqual(1, seed.count('"trusted_sources"'))

    def test_review_policy_is_marked_as_a_demo_choice(self) -> None:
        seed = self._seed()
        self.assertIn('"policy":"thresholds"', seed)
        self.assertIn("DEMO CHOICE, NOT A PRODUCTION DEFAULT", seed)

    def test_corpus_ships_with_the_case(self) -> None:
        corpus = sorted(path.name for path in Path("knowledge_base/corpus").glob("*.md"))
        self.assertEqual(["data-retention.md", "offline-mode.md", "team-roles.md"], corpus)
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(["corpus/*.md"], project["tool"]["setuptools"]["package-data"]["knowledge_base"])
        self.assertIn("knowledge", project["project"]["optional-dependencies"])

    def test_optional_stack_has_its_own_compose_file(self) -> None:
        compose = Path("docker/docker-compose.knowledge.yml").read_text(encoding="utf-8")
        self.assertIn("opensearchproject/opensearch", compose)
        self.assertIn("migrate-knowledge", compose)
        self.assertNotIn("qdrant", compose)


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
        self.assertIn('"severity": "high"', evidence_of(llm.calls[0]))
        self.assertIn('"selected_route": "urgent_response"', evidence_of(llm.calls[0]))
        self.assertNotIn("urgent_response", instructions_of(llm.calls[0]))

    async def test_invalid_metrics_take_manual_review_path(self) -> None:
        ctx = context("Triage alert", payload={"alert": {"error_rate": "not-a-number"}})
        normalized = await IncidentNormalizeStep(config={}).run(ctx)
        self.assertEqual("invalid", normalized.verdict)


if __name__ == "__main__":
    unittest.main()
