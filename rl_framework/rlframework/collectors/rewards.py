"""Reward resolvers used by decision collectors."""

from __future__ import annotations

from typing import Mapping

from rlframework.environment import AgentId, MultiEnvStep


class SharedTensorRewardResolver:
    """Read a per-agent reward vector on each tick, independently of discounting."""

    def __init__(
        self, field, agent_indices: Mapping[AgentId, int], scale: float = 1.0
    ) -> None:
        self.field, self.agent_indices, self.scale = (
            field,
            dict(agent_indices),
            float(scale),
        )

    def __call__(self, agent_id: AgentId, step: MultiEnvStep) -> float:
        if self.field not in step.shared.keys(True, True):
            return 0.0
        return (
            float(step.shared.get(self.field)[self.agent_indices[agent_id]].item())
            * self.scale
        )


def _active_reward(agent_id: AgentId, step: MultiEnvStep) -> float:
    return float(step.agents[agent_id].reward) if agent_id in step.agents else 0.0
