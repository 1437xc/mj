"""A compact discrete-action PPO reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as functional
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch import nn

from rlframework.data import SequenceBatch
from rlframework.policy import ActionEvaluation, ActorCriticPolicy, PolicyOutput
from rlframework.processing.sequence import training_view
from rlframework.utils.module import module_device





ObservationKey = Union[str, Tuple[str, ...]]


@dataclass(frozen=True)
class PPOConfig:
    """PPO optimization and GAE parameters for the reference learner."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True


class MLPActorCritic(nn.Module):
    """Shared MLP torso with categorical policy logits and a scalar critic."""

    def __init__(
        self, observation_size: int, action_count: int, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.torso = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_count)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return categorical logits and value estimates for a batch of observations."""
        features = self.torso(observation)
        return self.actor(features), self.critic(features).squeeze(-1)


class PPOPolicy(ActorCriticPolicy):
    """Actor-side categorical policy that persists action log-probability and value."""

    def __init__(
        self,
        module: MLPActorCritic,
        observation_key: str = "obs",
        action_mask_key: Optional[str] = "action_mask",
    ) -> None:
        super().__init__()
        self.module = module
        self.observation_key = observation_key
        self.action_mask_key = action_mask_key

    def act(
        self,
        observation: TensorDictBase,
        state: Optional[TensorDictBase] = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """Sample/choose one action and record values required by PPO training."""
        actor_observation = observation.to(module_device(self.module))
        vector = actor_observation.get(self.observation_key).float().unsqueeze(0)
        with torch.no_grad():
            logits, value = self.module(vector)
        logits = _masked_logits(
            logits.squeeze(0), actor_observation, self.action_mask_key
        )
        distribution = torch.distributions.Categorical(logits=logits)
        action = logits.argmax() if deterministic else distribution.sample()
        return PolicyOutput(
            action=TensorDict({"action": action.to(torch.int64)}, batch_size=[]),
            extras=TensorDict(
                {
                    "logp": distribution.log_prob(action).detach(),
                    "value": value.squeeze(0).detach(),
                },
                batch_size=[],
            ),
            next_state=state,
        )

    def values(
        self, observation: TensorDictBase, state=None, episode_start=None
    ) -> torch.Tensor:
        """Compute critic values for a batched feature TensorDict."""
        original_device = observation.get(self.observation_key).device
        model_observation = observation.to(module_device(self.module))
        _, values = self.module(model_observation.get(self.observation_key).float())
        return values.to(original_device)

    def evaluate_actions(
        self,
        observation: TensorDictBase,
        actions: TensorDictBase,
        state=None,
        episode_start=None,
    ) -> ActionEvaluation:
        """Return action log-probability, entropy and critic value for a batch."""
        logits, values = self.module(observation.get(self.observation_key).float())
        logits = _masked_logits(logits, observation, self.action_mask_key)
        distribution = torch.distributions.Categorical(logits=logits)
        return ActionEvaluation(
            distribution.log_prob(actions.get("action")), distribution.entropy(), values
        )


class PPOLearner:
    """Learner-side PPO update; trajectory processing remains a separate component."""

    def __init__(
        self,
        policy: ActorCriticPolicy,
        config: PPOConfig = PPOConfig(),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.policy = policy.to(device)
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )

    def update(self, batch) -> Mapping[str, float]:
        """Optimize flat transitions or entire sequences with fixed behavior terms.

        Advantage normalization uses valid training positions across the whole
        rollout. Sequence minibatches shuffle rows only; padding and burn-in
        never enter objectives, metrics or their denominators.
        """
        values = batch.to(self.device)
        data = values.data if isinstance(values, SequenceBatch) else values
        mask = (
            values.valid_mask
            if isinstance(values, SequenceBatch)
            else torch.ones(data.batch_size, dtype=torch.bool, device=self.device)
        )
        if not mask.any():
            raise ValueError("cannot train PPO without valid positions")
        if self.config.epochs <= 0 or self.config.minibatch_size <= 0:
            raise ValueError("epochs and minibatch_size must be positive")
        advantages = data.get("advantage").detach().float().clone()
        if self.config.normalize_advantage:
            valid = advantages[mask]
            advantages = (advantages - valid.mean()) / (
                valid.std(unbiased=False) + 1e-8
            )
        data = data.clone()
        data.set("advantage", advantages)
        if isinstance(values, SequenceBatch):
            values.data = data
        else:
            values = data
        totals, updates, positions = {}, 0, 0
        count = data.batch_size[0]
        # For sequences the configured size is an approximate number of tokens;
        # at least one intact sequence is always included.
        rows = (
            max(1, self.config.minibatch_size // data.batch_size[1])
            if isinstance(values, SequenceBatch)
            else self.config.minibatch_size
        )
        for _ in range(self.config.epochs):
            order = torch.randperm(count)
            for start in range(0, count, rows):
                index = order[start : start + rows]
                mini = (
                    values.select(index)
                    if isinstance(values, SequenceBatch)
                    else values[index.to(self.device)]
                )
                items, valid, kwargs = training_view(mini, self.policy)
                evaluation = self.policy.evaluate_actions(
                    items.get("obs"), items.get("action"), **kwargs
                )
                loss, metrics = ppo_objective(evaluation, items, valid, self.config)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                weight = int(valid.sum())
                metrics["grad_norm"] = grad_norm
                for name, value in metrics.items():
                    totals[name] = (
                        totals.get(name, 0.0) + float(value.detach().cpu()) * weight
                    )
                positions += weight
                updates += 1
        result = {"ppo/" + name: value / positions for name, value in totals.items()}
        result.update(
            {"ppo/updates": float(updates), "gradient_updates": float(updates)}
        )
        return result


def ppo_objective(evaluation, batch, mask, config):
    """Clipped PPO loss on valid positions, exposed for numerical verification."""
    for term in (evaluation.log_prob, evaluation.entropy, evaluation.value):
        if term.shape != mask.shape:
            raise ValueError("action evaluation shape must match the training batch")
    if not mask.any():
        raise ValueError("PPO objective requires valid positions")
    logp, entropy, value = (
        evaluation.log_prob[mask],
        evaluation.entropy[mask],
        evaluation.value[mask],
    )
    old_logp = batch.get(("policy", "extras", "logp"))[mask].detach()
    advantage, returns = (
        batch.get("advantage")[mask].detach(),
        batch.get("return")[mask].detach(),
    )
    log_ratio = logp - old_logp
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    value_loss = (value - returns).square().mean()
    entropy_bonus = entropy.mean()
    loss = (
        policy_loss
        + config.value_coefficient * value_loss
        - config.entropy_coefficient * entropy_bonus
    )
    return loss, {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_bonus,
        "approx_kl": ((ratio - 1) - log_ratio).mean(),
        "clip_fraction": ((ratio - 1).abs() > config.clip_epsilon).float().mean(),
        "ratio": ratio.mean(),
    }


def _masked_logits(
    logits: torch.Tensor, observation: TensorDictBase, key: Optional[str]
) -> torch.Tensor:
    """Apply an optional action mask without changing the policy interface."""
    if key is None or key not in observation.keys(True, True):
        return logits
    mask = observation.get(key).bool()
    if mask.shape != logits.shape:
        raise ValueError("PPO action_mask must match logits shape")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every PPO observation must permit at least one action")
    return logits.masked_fill(~mask, float("-inf"))
