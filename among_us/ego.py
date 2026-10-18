"""Cognitive profiles: how an agent attends, encodes, retains, judges and speaks.

``EgoProfile`` is the single definition of what an ego file may contain. Every
parameter is a number with a range and a unit that feeds exactly one formula;
no code branches on which ego it is running. A new ego is a new set of values,
a new behaviour is a new parameter here plus its one consumer.

Profiles ship with the package as ``egos/<ego_id>.toml``, validated strictly
(types, ranges, no unknown keys). They are deployment-authored, which is why a
persona may become trusted instructions; a request can only *select* a profile
by id, never supply one. ``egos/ego.schema.json`` is generated from this module
for editor validation (``python -m among_us.ego``).
"""

from __future__ import annotations

import json
import tomllib
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, Strict, ValidationError

from among_us.protocol import EventKind

#: What any profile notices regardless of temperament: a murder, a body, a meeting.
ALWAYS_NOTICED = frozenset({
    EventKind.KILL_WITNESSED, EventKind.BODY_FOUND, EventKind.MEETING_CALLED,
    EventKind.KILL_COMMITTED, EventKind.SABOTAGE_ALARM, EventKind.SHAPESHIFT_SEEN,
})

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
#: TOML keys are strings; an event kind key is parsed from its name, and an
#: unknown name is still refused.
EventKindKey = Annotated[EventKind, Strict(False)]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Attention(_Section):
    """Whether a perceived event is noticed at all (memory.encode)."""

    default: Probability = Field(description="Probability of noticing an event of any kind not listed in by_kind.")
    by_kind: dict[EventKindKey, Probability] = Field(
        default_factory=dict,
        description="Per event kind override of default. Kills, bodies and meetings are always noticed.",
    )


class Encoding(_Section):
    """How much of a noticed event is kept, and how faithfully (memory.encode)."""

    fidelity: Probability = Field(
        description="Probability of keeping each detail (room, actor, target, text); also the share of a chat "
        "message's words kept. Raised by 0.3 for kills and bodies.",
    )
    detail_noise: Probability = Field(
        description="Probability that a kept room is remembered as a neighbouring room instead.",
    )


class Retention(_Section):
    """How long a trace stays recallable (memory.strength, memory.recall)."""

    half_life_ticks: float = Field(
        gt=0, description="Ticks until a trace's strength halves. Doubled for kills and bodies.",
    )
    recall_threshold: Probability = Field(
        description="Minimum strength for a trace to come to mind. Below half of it, the trace is deleted.",
    )


class Judgement(_Section):
    """How recalled evidence turns into suspicion (beliefs.update_beliefs)."""

    suspicion_bias: float = Field(
        ge=0.0, le=3.0, description="Multiplier on incriminating evidence; exonerating evidence is not scaled.",
    )


class Voice(_Section):
    """How the agent speaks (decide prompt, validator)."""

    max_words: int = Field(ge=5, le=120, description="Upper bound on words per meeting statement.")


class EgoProfile(_Section):
    """One ego file. ``ego_id`` comes from the file name, never from the file."""

    ego_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1, max_length=40, description="Display name.")
    persona: str = Field(
        min_length=20, max_length=1200, description="Character description; becomes part of the trusted instructions.",
    )
    attention: Attention
    encoding: Encoding
    retention: Retention
    judgement: Judgement
    voice: Voice

    def attention_for(self, kind: EventKind) -> float:
        if kind in ALWAYS_NOTICED:
            return 1.0
        return self.attention.by_kind.get(kind, self.attention.default)


class UnknownEgo(KeyError):
    pass


class InvalidEgo(ValueError):
    pass


def _folder():
    return resources.files("among_us").joinpath("egos")


def ego_ids() -> tuple[str, ...]:
    return tuple(sorted(p.name.removesuffix(".toml") for p in _folder().iterdir() if p.name.endswith(".toml")))


def parse_ego(ego_id: str, text: str) -> EgoProfile:
    raw = tomllib.loads(text)
    if "ego_id" in raw:
        raise InvalidEgo(f"ego '{ego_id}': ego_id is the file name and must not appear in the file")
    try:
        return EgoProfile.model_validate({**raw, "ego_id": ego_id})
    except ValidationError as exc:
        raise InvalidEgo(f"ego '{ego_id}' is invalid:\n{exc}") from exc


@cache
def load_ego(ego_id: str) -> EgoProfile:
    if ego_id not in ego_ids():
        raise UnknownEgo(ego_id)
    return parse_ego(ego_id, _folder().joinpath(f"{ego_id}.toml").read_text(encoding="utf-8"))


def file_schema() -> dict:
    """JSON schema of an ego *file*: the profile without the file-name id."""

    schema = EgoProfile.model_json_schema()
    schema["properties"].pop("ego_id")
    schema["required"].remove("ego_id")
    schema["title"] = "nlght-ai Among Us ego profile"
    return schema


if __name__ == "__main__":
    target = Path(__file__).with_name("egos") / "ego.schema.json"
    target.write_text(json.dumps(file_schema(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")
