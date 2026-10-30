"""Recurrent categorical actor-critic with a policy-owned TensorDict state schema."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from rlframework.algorithm.distributions import masked_categorical
from rlframework.policy import ActionEvaluation, ActorCriticPolicy, PolicyOutput
from rlframework.utils.module import module_device


class GRUActorCritic(nn.Module):
    """Reference GRU cell; replace the policy to use another state or action tree."""

    def __init__(self, observation_size, action_count, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Linear(observation_size, hidden_size)
        self.cell = nn.GRUCell(hidden_size, hidden_size)
        self.actor, self.critic = nn.Linear(hidden_size, action_count), nn.Linear(
            hidden_size, 1
        )

    def forward(self, observation, hidden):
        hidden = self.cell(torch.tanh(self.encoder(observation)), hidden)
        return self.actor(hidden), self.critic(hidden).squeeze(-1), hidden


class RecurrentCategoricalPolicy(ActorCriticPolicy):
    """Accept independent [B] observations or ordered [B,T] sequences.

    Incoming state precedes the first observation; an episode_start flag clears
    it before that observation. Single-step acting and sequence learning share
    exactly the same cell path. Learners do not depend on the 'hidden' key.
    """

    def __init__(self, module, observation_key="obs", action_mask_key="action_mask"):
        super().__init__()
        self.module, self.observation_key, self.action_mask_key = (
            module,
            observation_key,
            action_mask_key,
        )

    def initial_state(self, observation):
        """Create the zero incoming state for one new environment episode."""
        return TensorDict(
            {
                "hidden": torch.zeros(
                    self.module.hidden_size, device=module_device(self)
                )
            },
            [],
        )

    def _unroll(self, observation, state=None, episode_start=None):
        device = module_device(self)
        obs = observation.get(self.observation_key).float().to(device)
        shape = tuple(observation.batch_size)
        if len(shape) not in (1, 2):
            raise ValueError("recurrent evaluation expects [B] or [B,T] observations")
        sequence = len(shape) == 2
        if not sequence:
            obs = obs.unsqueeze(1)
        hidden = (
            torch.zeros(obs.shape[0], self.module.hidden_size, device=device)
            if state is None
            else state.get("hidden").to(device)
        )
        reset = episode_start
        if reset is not None:
            reset = reset.to(device)
            if not sequence:
                reset = reset.unsqueeze(1)
        logits, values = [], []
        for time in range(obs.shape[1]):
            if reset is not None:
                hidden = torch.where(
                    reset[:, time].unsqueeze(-1), torch.zeros_like(hidden), hidden
                )
            action_logits, value, hidden = self.module(obs[:, time], hidden)
            logits.append(action_logits)
            values.append(value)
        logits, values = torch.stack(logits, 1), torch.stack(values, 1)
        return (
            logits if sequence else logits[:, 0],
            values if sequence else values[:, 0],
            hidden,
        )

    def act(self, observation, state=None, deterministic=False):
        """Advance the same recurrent cell used by sequence learning exactly once."""
        with torch.no_grad():
            logits, value, hidden = self._unroll(
                observation.unsqueeze(0), None if state is None else state.unsqueeze(0)
            )
            mask = (
                observation.get(self.action_mask_key, None)
                if self.action_mask_key
                else None
            )
            distribution = masked_categorical(
                logits[0], None if mask is None else mask.to(logits.device)
            )
            action = (
                distribution.probs.argmax(-1)
                if deterministic
                else distribution.sample()
            )
            return PolicyOutput(
                TensorDict({"action": action}, []),
                TensorDict(
                    {"logp": distribution.log_prob(action), "value": value[0]}, []
                ),
                TensorDict({"hidden": hidden[0]}, []),
            )

    def values(self, observation, state=None, episode_start=None):
        """Evaluate the critic without requiring a legal action mask at terminal frames."""
        return self._unroll(observation, state, episode_start)[1]

    def evaluate_actions(self, observation, actions, state=None, episode_start=None):
        """Evaluate independent observations or ordered sequences with joint action terms."""
        logits, value, _ = self._unroll(observation, state, episode_start)
        mask = (
            observation.get(self.action_mask_key, None)
            if self.action_mask_key
            else None
        )
        distribution = masked_categorical(
            logits, None if mask is None else mask.to(logits.device)
        )
        return ActionEvaluation(
            distribution.log_prob(actions.get("action").to(logits.device)),
            distribution.entropy(),
            value,
        )
