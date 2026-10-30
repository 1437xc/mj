"""REINFORCE components for discrete-action TensorDict trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import torch
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch import nn

from rlframework.policy import Policy, PolicyOutput
from rlframework.utils.module import module_device




ObservationKey = Union[str, Tuple[str, ...]]


@dataclass(frozen=True)
class PGConfig:
    """Optimization settings for vanilla Monte-Carlo policy gradient."""

    gamma: float = 0.99
    learning_rate: float = 3e-4
    entropy_coefficient: float = 0.0
    normalize_returns: bool = True


class MLPCategoricalPolicy(nn.Module):
    """A compact logits network suitable for CartPole and grid-world examples."""

    def __init__(
        self, observation_size: int, action_count: int, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return categorical logits for one or more observations."""
        return self.network(observation)


class PGPolicy(Policy):
    """Actor-side categorical policy that stores its sampled log probability."""

    def __init__(
        self,
        network: nn.Module,
        observation_key: str = "obs",
        action_mask_key: Optional[str] = "action_mask",
    ) -> None:
        super().__init__()
        self.network = network
        self.observation_key = observation_key
        self.action_mask_key = action_mask_key

    def act(
        self,
        observation: TensorDictBase,
        state: Optional[TensorDictBase] = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """Choose one valid action from an unbatched named observation."""
        actor_observation = observation.to(module_device(self.network))
        with torch.no_grad():
            logits = self.network(
                actor_observation.get(self.observation_key).float().unsqueeze(0)
            ).squeeze(0)
        logits = masked_logits(logits, actor_observation, self.action_mask_key)
        distribution = torch.distributions.Categorical(logits=logits)
        action = logits.argmax() if deterministic else distribution.sample()
        return PolicyOutput(
            action=TensorDict({"action": action.to(torch.int64)}, batch_size=[]),
            extras=TensorDict(
                {"logp": distribution.log_prob(action).detach()}, batch_size=[]
            ),
            next_state=state,
        )


@dataclass(frozen=True)
class MonteCarloReturnProcessor:
    """Add discounted episode returns to a one-dimensional trajectory TensorDict."""

    gamma: float = 0.99

    def process(self, trajectory: TensorDictBase) -> TensorDictBase:
        """Respect both terminated and truncated episode boundaries."""
        if len(trajectory.batch_size) != 1:
            raise ValueError(
                "MonteCarloReturnProcessor expects a one-dimensional time batch"
            )
        rewards = trajectory.get("reward").float()
        ended = trajectory.get("terminated").bool() | trajectory.get("truncated").bool()
        returns = torch.zeros_like(rewards)
        carry = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        for index in range(rewards.shape[0] - 1, -1, -1):
            carry = rewards[index] + self.gamma * carry * (1.0 - ended[index].float())
            returns[index] = carry
        result = trajectory.clone()
        result.set("return", returns)
        return result


class PGLearner:
    """REINFORCE learner; collection and return calculation stay outside the class."""

    def __init__(
        self,
        policy: PGPolicy,
        config: PGConfig = PGConfig(),
        observation_key: ObservationKey = ("obs", "obs"),
        action_key: ObservationKey = ("action", "action"),
        action_mask_key: Optional[ObservationKey] = ("obs", "action_mask"),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.policy = policy.to(device)
        self.config = config
        self.observation_key = observation_key
        self.action_key = action_key
        self.action_mask_key = action_mask_key
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )

    def update(self, batch: TensorDictBase) -> Mapping[str, float]:
        """Optimize one complete-return batch without any environment dependency."""
        values = batch.to(self.device)
        returns = values.get("return").float()
        if self.config.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        logits = self.policy.network(values.get(self.observation_key).float())
        logits = masked_logits(
            logits, values.get("obs"), _leaf_key(self.action_mask_key)
        )
        distribution = torch.distributions.Categorical(logits=logits)
        logp = distribution.log_prob(values.get(self.action_key).long())
        entropy = distribution.entropy().mean()
        loss = -(logp * returns).mean() - self.config.entropy_coefficient * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "gradient_updates": 1.0,
            "pg/loss": float(loss.detach().cpu()),
            "pg/return_mean": float(batch.get("return").mean()),
        }


def masked_logits(
    logits: torch.Tensor, observation: TensorDictBase, key: Optional[str]
) -> torch.Tensor:
    """Apply an optional per-action validity mask to logits of shape ``[..., A]``."""
    if key is None or key not in observation.keys(True, True):
        return logits
    mask = observation.get(key).bool()
    if mask.shape != logits.shape:
        raise ValueError("action_mask must have the same shape as categorical logits")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every categorical observation must allow at least one action")
    return logits.masked_fill(~mask, float("-inf"))


def _leaf_key(key: Optional[ObservationKey]) -> Optional[str]:
    """Convert the conventional nested observation mask key to its local name."""
    if key is None:
        return None
    if isinstance(key, str):
        return key
    return key[-1]
