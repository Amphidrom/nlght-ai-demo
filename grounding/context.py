"""Case evidence enters the model as typed, untrusted context records.

Every showcase case grounds its answer in something the runtime established
itself: a retrieved manual passage, a tool result, a deterministic policy
evaluation, a sandboxed report, a persisted preference. None of it is
deployment-authored, so none of it belongs in the trusted instruction channel.
These helpers keep that boundary cheap to honour in each step.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from nlght.core.model.messages import (
    CanonicalMessage,
    ContextKind,
    ContextProvenance,
    ContextRecord,
    UserMessage,
)


def last_user_text(messages: Sequence[CanonicalMessage]) -> str:
    """Return the most recent genuine user request as text.

    A ``CallerInstructionMessage`` is deliberately skipped: a client that sent
    ``role: system`` made a claim, it did not ask the question.
    """

    for message in reversed(list(messages)):
        if isinstance(message, UserMessage):
            return str(message.content)
    return ""


def evidence(
    *,
    record_id: str,
    kind: ContextKind,
    content: Any,
    source: str,
    **provenance: Any,
) -> ContextRecord:
    """Build one inert context record with explicit lineage.

    Non-string content is serialised as JSON so a record stays a single
    escaped payload rather than becoming prose the model could read as
    guidance. Unknown provenance keywords raise instead of being dropped.
    """

    return ContextRecord(
        record_id=record_id,
        kind=kind,
        content=content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
        provenance=ContextProvenance(source=source, **provenance),
    )
