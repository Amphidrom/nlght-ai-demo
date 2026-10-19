"""Which model each agent decides with, and which models exist.

Providers are the platform's configured model providers (``platform.yaml``). Each one has an nlght
provider backend (``ModelProviderBackend``) that lists the models it serves (``list_models()``), so
the choices offered are what the provider really has, not a list kept here.

An agent's model is resolved in this order:

1. the operator's choice for this agent in its lobby (admin UI),
2. its ego's ``[model]`` table,
3. the platform default — the first configured provider and its default model.

Meetings may use a model of their own — there is time to weigh what was said — chosen the same way
(``[model.meeting]``, a meeting choice in the admin UI); without one, the game model decides there too.

The decide step then runs on that provider and model: the turn's payload carries
``model_provider`` and ``model``, which the nlght executor binds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from among_us.ego import EgoProfile

logger = logging.getLogger(__name__)

#: Seconds a provider's model list is reused before it is asked again.
LISTING_TTL = 60.0


@dataclass(frozen=True)
class ModelChoice:
    provider: str
    #: ``None``: the provider's own default model.
    model: str | None
    #: "override" (admin UI), "ego" (the ego file) or "default" (the platform's).
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "source": self.source}


class ModelChoiceError(ValueError):
    """A provider or model the platform does not have."""


class ModelCatalog:
    def __init__(self, backends: dict[str, Any], default_models: dict[str, str | None]) -> None:
        #: provider name → nlght provider backend, in configuration order.
        self._backends = dict(backends)
        self._default_models = dict(default_models)
        self._listings: dict[str, tuple[float, list[str], str | None]] = {}

    @property
    def providers(self) -> list[str]:
        return list(self._backends)

    def default(self) -> ModelChoice | None:
        if not self._backends:
            return None
        provider = next(iter(self._backends))
        return ModelChoice(provider, self._default_models.get(provider), "default")

    def resolve(self, ego: EgoProfile, override: ModelChoice | None = None, *, meeting: bool = False) -> ModelChoice | None:
        if override is not None:
            return override
        if ego.model is not None:
            chosen = ego.model.meeting if meeting and ego.model.meeting is not None else ego.model
            return ModelChoice(chosen.provider, chosen.name or self._default_models.get(chosen.provider), "ego")
        return self.default()

    def usable(self, choice: ModelChoice | None) -> bool:
        """Whether a turn can run on this choice: its provider is configured here."""

        return choice is not None and choice.provider in self._backends

    async def models(self, provider: str) -> tuple[list[str], str | None]:
        """The models the provider serves (asked at most every ``LISTING_TTL`` seconds), and why the list is empty."""

        cached = self._listings.get(provider)
        if cached and time.monotonic() - cached[0] < LISTING_TTL:
            return cached[1], cached[2]
        backend = self._backends.get(provider)
        if backend is None:
            return [], f"provider '{provider}' is not configured"
        try:
            names, error = sorted({info.name for info in await backend.list_models()}), None
        except Exception as exc:  # A provider that is down must not take the catalog down.
            logger.warning("among_us.models.listing_failed | provider=%s | %s", provider, exc)
            names, error = [], str(exc) or type(exc).__name__
        if not names and error is None:
            error = "the provider listed no models"
        self._listings[provider] = (time.monotonic(), names, error)
        return names, error

    async def listing(self) -> list[dict[str, Any]]:
        result = []
        for provider in self._backends:
            names, error = await self.models(provider)
            result.append({"name": provider, "default_model": self._default_models.get(provider),
                           "models": names, "error": error})
        return result

    async def validate(self, provider: str, model: str | None) -> ModelChoice:
        """An operator's choice, checked against what the provider serves."""

        if provider not in self._backends:
            raise ModelChoiceError(f"Provider '{provider}' is not configured; known: {self.providers}.")
        names, error = await self.models(provider)
        if model is not None and error is None and model not in names:
            raise ModelChoiceError(f"Provider '{provider}' has no model '{model}'.")
        return ModelChoice(provider, model or self._default_models.get(provider), "override")


def catalog_from_container(container: Any) -> ModelCatalog:
    """The catalog of the running platform.

    nlght-ai 0.1.4 keeps the provider backends inside the workflow executor (``_model_clients``); the
    container has no public field for them yet. Configuration order comes from the runtime context.
    """

    clients: dict[str, Any] = getattr(container.workflow_executor, "_model_clients", None) or {}
    configured = list(container.runtime_context.model_providers)
    order = [p.name for p in configured if p.name in clients] + [n for n in clients if n not in {p.name for p in configured}]
    defaults = {p.name: getattr(p, "default_model", None) or None for p in configured}
    return ModelCatalog({name: clients[name] for name in order}, defaults)
