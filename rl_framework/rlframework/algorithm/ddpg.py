"""Deterministic continuous-control actor-critic components."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as functional
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch import nn

from rlframework.policy import Policy, PolicyOutput
from rlframework.utils.module import module_device




ObservationKey = Union[str, Tuple[str, ...]]


@dataclass(frozen=True)
class DDPGConfig:
    """DDPG target, exploration and optimization settings."""

    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    exploration_std: float = 0.1


class DDPGActor(nn.Module):
    """Deterministic actor whose output is normalized to the ``[-1, 1]`` box."""

    def __init__(
        self, observation_size: int, action_size: int, hidden_size: int = 128
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return normalized continuous actions."""
        return self.network(observation)


class DDPGCritic(nn.Module):
    """Q critic over an observation/action pair."""

    def __init__(
        self, observation_size: int, action_size: int, hidden_size: int = 128
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return one Q estimate per batch item."""
        return self.network(torch.cat((observation, action), dim=-1)).squeeze(-1)


class DDPGPolicy(Policy):
    """Actor-side deterministic policy with optional Gaussian exploration noise."""

    def __init__(
        self,
        actor: DDPGActor,
        exploration_std: float = 0.1,
        observation_key: str = "obs",
    ) -> None:
        super().__init__()
        self.actor = actor
        self.exploration_std = exploration_std
        self.observation_key = observation_key

    def act(
        self,
        observation: TensorDictBase,
        state: Optional[TensorDictBase] = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """Return a normalized continuous action tree for one observation."""
        actor_observation = observation.to(module_device(self.actor))
        with torch.no_grad():
            action = self.actor(
                actor_observation.get(self.observation_key).float().unsqueeze(0)
            ).squeeze(0)
        if not deterministic:
            action = action + torch.randn_like(action) * self.exploration_std
        action = action.clamp(-1.0, 1.0)
        return PolicyOutput(
            action=TensorDict({"action": action}, batch_size=[]),
            extras=TensorDict({}, batch_size=[]),
            next_state=state,
        )


class DDPGLearner:
    """Off-policy DDPG learner; replay ownership remains with the workflow."""

    def __init__(
        self,
        actor: DDPGActor,
        critic: DDPGCritic,
        config: DDPGConfig = DDPGConfig(),
        observation_key: ObservationKey = ("obs", "obs"),
        action_key: ObservationKey = ("action", "action"),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.target_actor = copy.deepcopy(actor).to(device)
        self.target_critic = copy.deepcopy(critic).to(device)
        self.config = config
        self.observation_key = observation_key
        self.action_key = action_key
        self.device = torch.device(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )

    def update(self, batch: TensorDictBase) -> Mapping[str, float]:
        """Apply one normalized-action DDPG update to an off-policy batch."""
        values = batch.to(self.device)
        observation = values.get(self.observation_key).float()
        next_observation = values.get(
            ("next",) + _as_tuple(self.observation_key)
        ).float()
        action = values.get(self.action_key).float()
        reward = values.get("reward").float()
        terminated = values.get("terminated").float()
        with torch.no_grad():
            target_q = self.target_critic(
                next_observation, self.target_actor(next_observation)
            )
            target = reward + self.config.gamma * (1.0 - terminated) * target_q
        critic_loss = functional.mse_loss(self.critic(observation, action), target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(observation, self.actor(observation)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        _soft_update(self.target_actor, self.actor, self.config.tau)
        _soft_update(self.target_critic, self.critic, self.config.tau)
        return {
            "gradient_updates": 2.0,
            "ddpg/actor_loss": float(actor_loss.detach().cpu()),
            "ddpg/critic_loss": float(critic_loss.detach().cpu()),
        }


def _as_tuple(key: ObservationKey) -> Tuple[str, ...]:
    return (key,) if isinstance(key, str) else key


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Move target parameters toward source parameters without replacing objects."""
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)
