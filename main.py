# Copyright (c) 2026 r4id4h. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Entrypoint for the six small nlght-ai showcase cases."""

from document_intake import IntakeExtractStep, IntakeRespondStep, IntakeValidateStep
from incident_triage import IncidentClassifyStep, IncidentNormalizeStep, IncidentRespondStep
from preference_assistant import PreferenceRememberStep, PreferenceRespondStep
from product_faq import ProductAnswerStep, ProductRetrieveStep
from runtime_data_check import RuntimeCsvCheckStep, RuntimeCsvValidateStep
from shipping_quote import ShippingQuoteStep, ShippingRateTool, ShippingValidateStep
from showcase import ShowcaseAdapter

from nlght.adapters.inbound.registry import http_adapter_registry
from nlght.adapters.outbound.tools.registry import tool_registry
from nlght.adapters.outbound.workflow.registry import step_registry
from nlght.main import serve


for step in (
    ProductRetrieveStep,
    ProductAnswerStep,
    ShippingValidateStep,
    ShippingQuoteStep,
    IncidentNormalizeStep,
    IncidentClassifyStep,
    IncidentRespondStep,
    IntakeExtractStep,
    IntakeValidateStep,
    IntakeRespondStep,
    PreferenceRememberStep,
    PreferenceRespondStep,
    RuntimeCsvValidateStep,
    RuntimeCsvCheckStep,
):
    step_registry.register(step)

tool_registry.register(ShippingRateTool)
http_adapter_registry.register(ShowcaseAdapter())


def run() -> None:
    """Start the packaged demo application."""
    serve()


if __name__ == "__main__":
    run()
