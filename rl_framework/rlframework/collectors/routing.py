"""Static per-agent policy routing and behavior-version tracking."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from rlframework.collector import PolicyRef
from rlframework.environment import AgentId, AgentStep
from rlframework.policy import Policy


class StaticPolicyRouter:
    """Map agents to policies. Policy identity stays fixed throughout an episode."""

    def __init__(self, routes: Mapping[AgentId, PolicyRef]) -> None:
        self._routes = dict(routes)
        identities = {}
        for reference in self._routes.values():
            previous = identities.setdefault(reference.policy_id, reference.policy)
            if previous is not reference.policy:
                raise ValueError(
                    "one policy id must refer to one policy instance within a router"
                )

    @property
    def policies(self) -> Mapping[str, Policy]:
        """Expose optimizer ownership without exposing mutable routing entries."""
        return {
            reference.policy_id: reference.policy for reference in self._routes.values()
        }

    def resolve(self, agent_id: AgentId, step: AgentStep) -> PolicyRef:
        """Return the policy controlling an active agent; missing routes are errors."""
        return self._routes[agent_id]

    def set_version(self, policy_id: str, version: int) -> None:
        """Tag subsequent decisions after a workflow publishes new weights."""
        for agent, reference in self._routes.items():
            if reference.policy_id == policy_id:
                self._routes[agent] = replace(reference, version=version)

    @property
    def versions(self) -> Mapping[str, int]:
        """Published behavior version for each unique routed policy."""
        result = {}
        for reference in self._routes.values():
            previous = result.setdefault(reference.policy_id, reference.version)
            if previous != reference.version:
                raise ValueError("shared policy routes must have one behavior version")
        return result
