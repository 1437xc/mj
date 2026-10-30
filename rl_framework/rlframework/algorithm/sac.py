"""Compact continuous Soft Actor-Critic components using TensorDict batches."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as functional
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch import nn

from rlframework.algorithm.ddpg import DDPGCritic
from rlframework.policy import Policy, PolicyOutput
from rlframework.utils.module import module_device




ObservationKey = Union[str, Tuple[str, ...]]


@dataclass(frozen=True)
class SACConfig:
    """Fixed-temperature SAC hyperparameters for normalized continuous actions."""

    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4


class SACActor(nn.Module):
    """Tanh-squashed diagonal Gaussian policy over a normalized action box."""

    def __init__(
        self, observation_size: int, action_size: int, hidden_size: int = 128
    ) -> None:
        super().__init__()
        self.torso = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and clamped log standard deviation."""
        features = self.torso(observation)
        return self.mean(features), self.log_std(features).clamp(-20.0, 2.0)

    def sample(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterize an action and include the tanh change-of-variables term."""
        mean, log_std = self(observation)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        logp = distribution.log_prob(raw_action).sum(dim=-1)
        logp = logp - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, logp

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the mean action under the same normalized action contract."""
        mean, _ = self(observation)
        return torch.tanh(mean)


class SACPolicy(Policy):
    """Actor-side stochastic SAC policy with no replay or optimizer knowledge."""

    def __init__(self, actor: SACActor, observation_key: str = "obs") -> None:
        super().__init__()
        self.actor = actor
        self.observation_key = observation_key

    def act(
        self,
        observation: TensorDictBase,
        state: Optional[TensorDictBase] = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """Produce one action and record log probability for diagnostics."""
        actor_observation = observation.to(module_device(self.actor))
        vector = actor_observation.get(self.observation_key).float().unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(vector).squeeze(0)
                logp = torch.zeros((), dtype=action.dtype, device=action.device)
            else:
                action, logp = self.actor.sample(vector)
                action, logp = action.squeeze(0), logp.squeeze(0)
        return PolicyOutput(
            action=TensorDict({"action": action}, batch_size=[]),
            extras=TensorDict({"logp": logp}, batch_size=[]),
            next_state=state,
        )


class SACLearner:
    """Twin-critic fixed-temperature SAC learner for normalized action spaces."""

    def __init__(
        self,
        actor: SACActor,
        critic_one: DDPGCritic,
        critic_two: DDPGCritic,
        config: SACConfig = SACConfig(),
        observation_key: ObservationKey = ("obs", "obs"),
        action_key: ObservationKey = ("action", "action"),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.actor = actor.to(device)
        self.critic_one = critic_one.to(device)
        self.critic_two = critic_two.to(device)
        self.target_critic_one = copy.deepcopy(critic_one).to(device)
        self.target_critic_two = copy.deepcopy(critic_two).to(device)
        self.config = config
        self.observation_key = observation_key
        self.action_key = action_key
        self.device = torch.device(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_one.parameters()) + list(self.critic_two.parameters()),
            lr=config.critic_learning_rate,
        )

    def update(self, batch: TensorDictBase) -> Mapping[str, float]:
        """Apply twin-Q critic and entropy-regularized actor updates."""
        values = batch.to(self.device)
        observation = values.get(self.observation_key).float()
        next_observation = values.get(
            ("next",) + _as_tuple(self.observation_key)
        ).float()
        action = values.get(self.action_key).float()
        reward = values.get("reward").float()
        terminated = values.get("terminated").float()
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_observation)
            next_q = torch.min(
                self.target_critic_one(next_observation, next_action),
                self.target_critic_two(next_observation, next_action),
            )
            target = reward + self.config.gamma * (1.0 - terminated) * (
                next_q - self.config.alpha * next_logp
            )
        critic_loss = functional.mse_loss(self.critic_one(observation, action), target)
        critic_loss = critic_loss + functional.mse_loss(
            self.critic_two(observation, action), target
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        sampled_action, logp = self.actor.sample(observation)
        actor_q = torch.min(
            self.critic_one(observation, sampled_action),
            self.critic_two(observation, sampled_action),
        )
        actor_loss = (self.config.alpha * logp - actor_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        _soft_update(self.target_critic_one, self.critic_one, self.config.tau)
        _soft_update(self.target_critic_two, self.critic_two, self.config.tau)
        return {
            "gradient_updates": 2.0,
            "sac/actor_loss": float(actor_loss.detach().cpu()),
            "sac/critic_loss": float(critic_loss.detach().cpu()),
        }


def _as_tuple(key: ObservationKey) -> Tuple[str, ...]:
    return (key,) if isinstance(key, str) else key


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Perform a Polyak target update without coupling it to a workflow."""
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)
