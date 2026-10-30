"""Multi-agent environment interface and step payloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase



AgentId = str


@dataclass
class AgentStep:
    """One agent's result after reset or after an environment step."""

    observation: TensorDictBase
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class MultiEnvStep:
    """A set of agent steps plus optional shared information for central critics."""

    agents: Mapping[AgentId, AgentStep]
    episode_done: bool
    shared: TensorDictBase = field(
        default_factory=lambda: TensorDict({}, batch_size=[])
    )


class MultiEnv(ABC):
    """Minimal parallel multi-agent environment interface.

    ``step`` receives one action tree per active agent. Synchronous collectors
    require a result for every acted agent; sparse collectors wait for the next
    observation. Individual termination does not imply global episode end.
    Time-limit endings must provide true final observations for bootstrapping.
    """

    @abstractmethod
    def reset(self, config: Optional[Mapping[str, Any]] = None) -> MultiEnvStep:
        """Start an episode and return initial observations for active agents."""

    @abstractmethod
    def step(self, actions: Mapping[AgentId, TensorDictBase]) -> MultiEnvStep:
        """Advance one environment tick using an action tree per active agent."""
