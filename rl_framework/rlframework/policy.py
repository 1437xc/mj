"""Abstract policy contracts and action/evaluation payloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch import nn


@dataclass
class PolicyOutput:
    """Action plus only the extra fields a learner explicitly needs."""

    action: TensorDictBase
    extras: TensorDictBase = field(
        default_factory=lambda: TensorDict({}, batch_size=[])
    )
    next_state: Optional[TensorDictBase] = None


@dataclass
class ActionEvaluation:
    """Joint action terms with exactly the observation batch shape, [B] or [B,T].

    Policies own action structure and legality. Learners never cast or select
    action leaves and therefore work with categorical or conditional actions.
    """

    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class Policy(nn.Module, ABC):
    """Actor-side decision component; it never owns replay or optimization."""

    def initial_state(self, observation: TensorDictBase) -> Optional[TensorDictBase]:
        """Return a new episode's incoming state, or None for stateless policies.

        Recurrent policies implement this explicitly so collectors never infer
        initial values from a state that has already advanced.
        """
        return None

    @abstractmethod
    def act(
        self,
        observation: TensorDictBase,
        state: Optional[TensorDictBase] = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """Select one unbatched environment action and optional next RNN state."""


class ActorCriticPolicy(Policy, ABC):
    """Policy contract shared by actor-critic learners and return processors.

    The contract deliberately says nothing about network architecture or the
    observation fields consumed by actor and critic.  A policy may therefore
    implement a shared MLP, separate towers, or centralized training with a
    decentralized actor while the PPO/A2C learners stay unchanged.
    """

    @abstractmethod
    def values(
        self,
        observation: TensorDictBase,
        state: Optional[TensorDictBase] = None,
        episode_start: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate observations, using incoming RNN state and reset flags if supplied."""

    @abstractmethod
    def evaluate_actions(
        self,
        observation: TensorDictBase,
        actions: TensorDictBase,
        state: Optional[TensorDictBase] = None,
        episode_start: Optional[torch.Tensor] = None,
    ) -> ActionEvaluation:
        """Evaluate complete actions; sequences must preserve their time axis."""
