"""The perception boundary: an agent takes in its own observation, nothing else.

Visibility is filtered at the source (simulation or the nlght Impostor server). This module is
the second wall: it refuses an observation addressed to someone else, one
that discloses a role other than the agent's own, or an event the agent could
not have perceived. A violation raises — a leak is a defect, not noise.
"""

from __future__ import annotations

from among_us.protocol import EventKind, Observation, ObservedEvent, Role


class PerceptionViolation(ValueError):
    pass


def perceive(observation: Observation, agent_id: str) -> tuple[ObservedEvent, ...]:
    if observation.agent_id != agent_id or observation.self_state.player_id != agent_id:
        raise PerceptionViolation(f"observation for {observation.agent_id!r} delivered to {agent_id!r}")
    if not observation.self_state.alive:
        raise PerceptionViolation("a dead player is not an agent")
    if observation.self_state.role is Role.CREWMATE and observation.self_state.known_impostors:
        raise PerceptionViolation("a crewmate observation names impostors")
    if observation.self_state.role is Role.CREWMATE and (
        observation.self_state.special_role or observation.self_state.disguised_as or observation.self_state.vanished
    ):
        raise PerceptionViolation("a crewmate observation carries impostor abilities")
    seen: set[str] = set()
    for event in observation.events:
        if event.event_id in seen:
            raise PerceptionViolation(f"event {event.event_id} delivered twice")
        seen.add(event.event_id)
        if event.tick > observation.tick:
            raise PerceptionViolation(f"event {event.event_id} is from the future")
        if event.kind is EventKind.KILL_WITNESSED and event.actor == agent_id:
            raise PerceptionViolation("a killer does not witness their own kill")
        if event.kind is EventKind.KILL_COMMITTED and (
            event.actor != agent_id or observation.self_state.role is not Role.IMPOSTOR
        ):
            raise PerceptionViolation("only the killer remembers a kill as committed")
    return observation.events
