"""Cognitive profiles: how an agent attends, encodes, retains and judges.

Profiles ship with the package as TOML files. They are deployment-authored,
which is why a profile's persona may become trusted instructions; a request
can only *select* a profile by id, never supply one.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import cache
from importlib import resources

from among_us.protocol import EventKind

#: What any profile notices regardless of temperament: a body and a murder.
_ALWAYS_NOTICED = frozenset({EventKind.KILL_WITNESSED, EventKind.BODY_FOUND, EventKind.MEETING_CALLED})


@dataclass(frozen=True)
class EgoProfile:
    ego_id: str
    name: str
    persona: str
    attention_default: float
    encoding_fidelity: float
    retention_half_life: float
    recall_threshold: float
    detail_noise: float
    suspicion_bias: float
    max_words: int
    attention: dict[str, float] = field(default_factory=dict)

    def attention_for(self, kind: EventKind) -> float:
        if kind in _ALWAYS_NOTICED:
            return 1.0
        return self.attention.get(kind.value, self.attention_default)


class UnknownEgo(KeyError):
    pass


def ego_ids() -> tuple[str, ...]:
    folder = resources.files("among_us").joinpath("egos")
    return tuple(sorted(p.name.removesuffix(".toml") for p in folder.iterdir() if p.name.endswith(".toml")))


@cache
def load_ego(ego_id: str) -> EgoProfile:
    if ego_id not in ego_ids():
        raise UnknownEgo(ego_id)
    raw = tomllib.loads(resources.files("among_us").joinpath("egos", f"{ego_id}.toml").read_text(encoding="utf-8"))
    attention = {str(k): float(v) for k, v in dict(raw.pop("attention", {})).items()}
    unknown = set(attention) - {kind.value for kind in EventKind}
    if unknown:
        raise ValueError(f"ego '{ego_id}': unknown attention kinds {sorted(unknown)}")
    profile = EgoProfile(ego_id=ego_id, attention=attention, **raw)
    for name in ("attention_default", "encoding_fidelity", "recall_threshold", "detail_noise"):
        value = getattr(profile, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"ego '{ego_id}': {name} must be within [0, 1], got {value}")
    if profile.retention_half_life <= 0 or profile.max_words <= 0:
        raise ValueError(f"ego '{ego_id}': retention_half_life and max_words must be positive")
    return profile
