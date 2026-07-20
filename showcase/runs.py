"""Small persistent run observer used by the showcase UI."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nlght.core.signals.signal import Signal


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class RunRecord:
    correlation_id: str
    case_id: str
    workflow: str
    session_key: str | None = None
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    events: list[dict[str, Any]] = field(default_factory=list)
    result: str = ""
    error: str | None = None
    _changed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def public_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        value = {
            "correlation_id": self.correlation_id,
            "case_id": self.case_id,
            "workflow": self.workflow,
            "session_key": self.session_key,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
        }
        if include_events:
            value["events"] = list(self.events)
        return value


class RunStore:
    """In-process live store with JSON snapshots for completed demo runs."""

    def __init__(self, root: Path | str = ".demo-runs") -> None:
        self._root = Path(root)
        self._runs: dict[str, RunRecord] = {}

    def create(
        self,
        *,
        correlation_id: str,
        case_id: str,
        workflow: str,
        session_key: str | None = None,
    ) -> RunRecord:
        record = RunRecord(
            correlation_id=correlation_id,
            case_id=case_id,
            workflow=workflow,
            session_key=session_key,
        )
        self._runs[correlation_id] = record
        return record

    def get(self, correlation_id: str) -> RunRecord | None:
        record = self._runs.get(correlation_id)
        if record is not None:
            return record
        path = self._root / f"{correlation_id}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        record = RunRecord(
            correlation_id=data["correlation_id"],
            case_id=data["case_id"],
            workflow=data["workflow"],
            session_key=data.get("session_key"),
            status=data.get("status", "failed"),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
            events=list(data.get("events", [])),
            result=data.get("result", ""),
            error=data.get("error"),
        )
        self._runs[correlation_id] = record
        return record

    def set_status(self, record: RunRecord, status: str, *, error: str | None = None) -> None:
        record.status = status
        record.error = error
        record.updated_at = _now()
        record._changed.set()
        if status in {"succeeded", "failed", "cancelled"}:
            self._persist(record)

    def append(self, record: RunRecord, signal: Signal) -> None:
        event = {
            "sequence": len(record.events),
            "kind": signal.kind,
            "role": signal.role,
            "content": signal.content,
            "created_at": _now(),
        }
        record.events.append(event)
        if signal.kind in {"token", "result"} and signal.role == "assistant":
            record.result += signal.content
        record.updated_at = event["created_at"]
        record._changed.set()

    async def wait_for_change(self, record: RunRecord, timeout: float = 15.0) -> None:
        record._changed.clear()
        try:
            await asyncio.wait_for(record._changed.wait(), timeout=timeout)
        except TimeoutError:
            pass

    def _persist(self, record: RunRecord) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            path = self._root / f"{record.correlation_id}.json"
            path.write_text(json.dumps(record.public_dict(), indent=2), encoding="utf-8")
        except OSError:
            # The live showcase must continue even when its optional history path is read-only.
            return
