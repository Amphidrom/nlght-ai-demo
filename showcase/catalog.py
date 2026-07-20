"""Declarative catalog for the public showcase gallery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ShowcaseCase:
    id: str
    title: str
    tagline: str
    story: str
    workflow: str
    capabilities: tuple[str, ...]
    sample_input: dict[str, Any]
    examples: tuple[dict[str, Any], ...] = ()
    optional_integrations: tuple[str, ...] = ()
    session: bool = False
    advanced: bool = False

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


CASES: tuple[ShowcaseCase, ...] = (
    ShowcaseCase(
        id="product-faq",
        title="Product FAQ",
        tagline="Answer from three controlled manual articles with explicit source IDs.",
        story="Retrieve only matching Northstar Desk manual passages and refuse unsupported questions.",
        workflow="product_faq",
        capabilities=("grounding", "retrieval", "source IDs", "unknown handling"),
        sample_input={"message": "How long can I recover a deleted Northstar Desk project?"},
        examples=(
            {"title": "Supported question", "description": "Matches the retention article and must cite MANUAL-204.", "input": {"message": "How long can I recover a deleted Northstar Desk project?"}},
            {"title": "Combined question", "description": "Retrieves two manual articles and shows both source IDs.", "input": {"message": "Can I work offline and which team roles are available?"}},
            {"title": "Unsupported question", "description": "No article supports the answer, so the workflow must refuse.", "input": {"message": "Does Northstar Desk support satellite telemetry?"}},
        ),
    ),
    ShowcaseCase(
        id="shipping-quote",
        title="Shipping Quote",
        tagline="Validate a parcel and calculate one reproducible rate.",
        story="Pass validated shipment data to a registered custom tool and return its JSON result.",
        workflow="shipping_quote",
        capabilities=("custom tool", "validation", "tool catalog", "structured result"),
        sample_input={"shipment": {"weight_kg": 3.5, "zone": "eu", "express": True}},
        examples=(
            {"title": "EU express", "description": "Valid input reaches the custom rate tool.", "input": {"shipment": {"weight_kg": 3.5, "zone": "eu", "express": True}}},
            {"title": "Invalid parcel", "description": "A negative weight is rejected before any tool executes.", "input": {"shipment": {"weight_kg": -2, "zone": "eu", "express": False}}},
        ),
    ),
    ShowcaseCase(
        id="incident-triage",
        title="Incident Triage",
        tagline="Classify one operational signal and recommend the next action.",
        story=(
            "Normalize a service alert, apply transparent severity thresholds, and route "
            "it to an urgent response or a routine review."
        ),
        workflow="incident_triage",
        capabilities=("structured signals", "workflow branching", "validation", "status events"),
        sample_input={
            "alert": {"service": "checkout", "error_rate": 0.18, "p95_ms": 2400},
        },
        examples=(
            {"title": "Urgent route", "description": "High error rate and latency route to urgent_response.", "input": {"alert": {"service": "checkout", "error_rate": 0.18, "p95_ms": 2400}}},
            {"title": "Routine route", "description": "Healthy metrics route to manual_review.", "input": {"alert": {"service": "catalog", "error_rate": 0.01, "p95_ms": 320}}},
        ),
    ),
    ShowcaseCase(
        id="document-intake",
        title="Policy-Aware Document Intake",
        tagline="Extract structured routing data from incoming documents.",
        story=(
            "Extract document-specific fields, evaluate a named routing policy, and show "
            "why the intake was accepted, held for review, or escalated."
        ),
        workflow="document_intake",
        capabilities=("structured extraction", "validation", "workflow routing", "artifacts"),
        sample_input={
            "message": "Today Acme Ltd submitted a German renewal request for contract AC-1042."
        },
        examples=(
            {"title": "Complete renewal", "description": "Organization and contract ID satisfy the renewal policy.", "input": {"message": "Today Acme Ltd submitted a German renewal request for contract AC-1042."}},
            {"title": "Urgent threat bulletin", "description": "A governmental threat bulletin targeting Linux is escalated immediately.", "input": {"message": "Today the governmental cybersecurity agency released a report on an immediate threat against all Linux based infrastructure."}},
            {"title": "Incomplete intake", "description": "A generic note without an organization is routed to needs_review.", "input": {"message": "A new document arrived today and should be processed."}},
        ),
    ),
    ShowcaseCase(
        id="preference-assistant",
        title="Preference Assistant",
        tagline="Remember a preference, then apply it on a later request.",
        story="Run step 1 and step 2 with the same session key to see persisted memory change the result.",
        workflow="preference_assistant",
        capabilities=("Hive Mind", "session memory", "fact integrity", "session isolation"),
        sample_input={"session_key": "preference-demo", "action": "remember", "preference": {"response_style": "concise"}},
        examples=(
            {"title": "Step 1 · Remember", "description": "Stores a user-supplied response style in session preference-demo.", "input": {"session_key": "preference-demo", "action": "remember", "preference": {"response_style": "concise"}}},
            {"title": "Step 2 · Apply", "description": "Uses the same session and applies the remembered style to a new task.", "input": {"session_key": "preference-demo", "action": "apply", "task": "Explain why backups should be tested regularly."}},
            {"title": "Different session", "description": "Uses an isolated session and therefore cannot see the saved preference.", "input": {"session_key": "another-session", "action": "apply", "task": "Explain why backups should be tested regularly."}},
        ),
        session=True,
    ),
    ShowcaseCase(
        id="runtime-data-check",
        title="Runtime Data Check",
        tagline="Inspect a small CSV with one fixed command in the configured runtime.",
        story="Validate input and return row and missing-value counts from an isolated process.",
        workflow="runtime_data_check",
        capabilities=("OS runtime", "isolated process", "fixed command", "structured report"),
        sample_input={"csv": "name,email\nAda,ada@example.test\nGrace,"},
        examples=(
            {"title": "Missing value", "description": "Reports two rows and one missing email.", "input": {"csv": "name,email\nAda,ada@example.test\nGrace,"}},
            {"title": "Complete data", "description": "Reports zero missing values.", "input": {"csv": "name,email\nAda,ada@example.test\nGrace,grace@example.test"}},
        ),
    ),
)

CASE_BY_ID = {case.id: case for case in CASES}


def get_case(case_id: str) -> ShowcaseCase | None:
    return CASE_BY_ID.get(case_id)
