"""Gym environment adaptation to the multi-agent step contract."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any, Dict, Mapping, Optional

from tensordict.tensordict import TensorDictBase

from rlframework.environment import AgentId, AgentStep, MultiEnv, MultiEnvStep
from rlframework.feature import FeatureAdapter
from rlframework.utils.tensor import as_tensordict


class GymMultiEnvAdapter(MultiEnv):
    """Adapt a Gym or Gymnasium single-agent environment into ``MultiEnv``.

    No Gym package is imported here.  The adapter accepts both the historical
    four-value step result and Gymnasium's five-value result, which keeps the
    framework dependency-light and makes version handling explicit.
    """

    agent_id = "agent_0"

    def __init__(
        self, environment: Any, feature_adapter: Optional[FeatureAdapter] = None
    ) -> None:
        self._environment = environment
        self._feature_adapter = feature_adapter

    def reset(self, config: Optional[Mapping[str, Any]] = None) -> MultiEnvStep:
        kwargs: Dict[str, Any] = {}
        if config:
            options = dict(config)
            if "seed" in options:
                kwargs["seed"] = options.pop("seed")
            if options:
                kwargs["options"] = options
        result = self._environment.reset(**kwargs)
        observation = (
            result[0] if isinstance(result, tuple) and len(result) == 2 else result
        )
        return MultiEnvStep(
            {self.agent_id: AgentStep(self._encode(observation))}, episode_done=False
        )

    def step(self, actions: Mapping[AgentId, TensorDictBase]) -> MultiEnvStep:
        if set(actions) != {self.agent_id}:
            raise KeyError(
                "GymMultiEnvAdapter expects one action for {0}".format(self.agent_id)
            )
        result = self._environment.step(_to_environment_action(actions[self.agent_id]))
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            observation, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            raise ValueError("environment.step must return four or five values")
        step = AgentStep(
            observation=self._encode(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )
        return MultiEnvStep(
            {self.agent_id: step}, episode_done=step.terminated or step.truncated
        )

    def close(self) -> None:
        """Release resources owned by the wrapped environment."""
        self._environment.close()

    def _encode(self, observation: Any) -> TensorDictBase:
        if self._feature_adapter is not None:
            return self._feature_adapter.encode(observation)
        if isinstance(observation, MappingABC):
            return as_tensordict(observation)
        return as_tensordict({"obs": observation})


def _to_environment_action(action: TensorDictBase) -> Any:
    """Turn the conventional ``action`` leaf into a scalar/array for Gym."""
    if "action" not in action.keys():
        raise KeyError("a Gym action TensorDict must contain the 'action' field")
    value = action.get("action").detach().cpu()
    if value.ndim == 0:
        return value.item()
    return value.numpy()
