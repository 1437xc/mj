"""Synchronous advantage actor-critic built on PPO's reusable actor-critic policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

import torch
import torch.nn.functional as functional
from tensordict.tensordict import TensorDictBase

from rlframework.policy import ActorCriticPolicy
from rlframework.processing.sequence import training_view


@dataclass(frozen=True)
class A2CConfig:
    """A2C loss and optimizer settings shared with the A3C parameter server."""

    learning_rate: float = 3e-4
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 1.0


class A2CLearner:
    """Synchronous actor-critic learner using GAEProcessor-produced batches."""

    def __init__(
        self,
        policy: ActorCriticPolicy,
        config: A2CConfig = A2CConfig(),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.policy = policy.to(device)
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )

    def update(self, batch: TensorDictBase) -> Mapping[str, float]:
        """Apply one A2C update to a fresh on-policy TensorDict batch."""
        values = batch.to(self.device)
        loss, metrics = a2c_objective(self.policy, values, self.config)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        return dict(
            {name: float(value.detach().cpu()) for name, value in metrics.items()},
            gradient_updates=1.0,
        )


def a2c_objective(
    policy: ActorCriticPolicy, batch: TensorDictBase, config: A2CConfig
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Compute the reusable A2C objective without mutating an optimizer."""
    batch, mask, kwargs = training_view(batch, policy)
    evaluation = policy.evaluate_actions(
        batch.get("obs"), batch.get("action"), **kwargs
    )
    returns = batch.get("return")[mask].float()
    logp, entropy, predicted_value = (
        evaluation.log_prob[mask],
        evaluation.entropy[mask],
        evaluation.value[mask],
    )
    advantages = (
        batch.get("advantage")[mask].float()
        if "advantage" in batch.keys()
        else returns - predicted_value.detach()
    )
    policy_loss = -(logp * advantages.detach()).mean()
    value_loss = functional.mse_loss(predicted_value, returns)
    entropy_bonus = entropy.mean()
    loss = (
        policy_loss
        + config.value_coefficient * value_loss
        - config.entropy_coefficient * entropy_bonus
    )
    return loss, {
        "a2c/loss": loss,
        "a2c/policy_loss": policy_loss,
        "a2c/value_loss": value_loss,
        "a2c/entropy": entropy_bonus,
    }
