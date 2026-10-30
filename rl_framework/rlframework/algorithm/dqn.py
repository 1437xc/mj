"""A compact DQN reference implementation using TensorDict transition batches."""

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
class DQNConfig:
    """Hyperparameters that change DQN behavior rather than network architecture."""

    gamma: float = 0.99
    learning_rate: float = 1e-3
    target_update_interval: int = 100
    epsilon: float = 0.1


class MLPQNetwork(nn.Module):
    """Small Q network used by reference examples; users may supply any module instead."""

    def __init__(
        self, observation_size: int, action_count: int, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return one Q value per discrete action."""
        return self.network(observation)


class DQNPolicy(Policy):
    """Actor-side epsilon-greedy policy for a Q network and optional action mask."""

    def __init__(
        self,
        network: nn.Module,
        action_count: int,
        epsilon: float = 0.1,
        observation_key: str = "obs",
        action_mask_key: Optional[str] = "action_mask",
    ) -> None:
        super().__init__()
        self.network = network
        self.action_count = action_count
        self.epsilon = epsilon
        self.observation_key = observation_key
        self.action_mask_key = action_mask_key

    def act(
        self,
        observation: TensorDictBase,
        state: Optional[TensorDictBase] = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """Sample a valid discrete action from one unbatched observation TensorDict."""
        actor_observation = observation.to(module_device(self.network))
        vector = actor_observation.get(self.observation_key).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.network(vector).squeeze(0)
        mask = _mask_from_observation(
            actor_observation, self.action_mask_key, self.action_count
        )
        masked_q = _apply_mask(q_values, mask)
        explore = (
            not deterministic and torch.rand((), device=q_values.device) < self.epsilon
        )
        if explore:
            candidates = torch.nonzero(mask, as_tuple=False).flatten()
            action = candidates[torch.randint(len(candidates), (1,))].squeeze(0)
        else:
            action = masked_q.argmax(dim=-1)
        return PolicyOutput(
            action=TensorDict({"action": action.to(torch.int64)}, batch_size=[]),
            extras=TensorDict({"q_value": masked_q.max().detach()}, batch_size=[]),
            next_state=state,
        )


class DQNLearner:
    """Learner-side DQN update with a target network and no environment dependency."""

    def __init__(
        self,
        network: nn.Module,
        config: DQNConfig = DQNConfig(),
        observation_key: ObservationKey = ("obs", "obs"),
        action_key: ObservationKey = ("action", "action"),
        action_mask_key: Optional[ObservationKey] = ("obs", "action_mask"),
        next_action_mask_key: Optional[ObservationKey] = ("next", "obs", "action_mask"),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.network = network.to(device)
        self.target_network = copy.deepcopy(network).to(device)
        self.target_network.eval()
        self.config = config
        self.observation_key = observation_key
        self.action_key = action_key
        self.action_mask_key = action_mask_key
        self.next_action_mask_key = next_action_mask_key
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )
        self.update_count = 0

    def update(self, batch: TensorDictBase) -> Mapping[str, float]:
        """Perform one DQN gradient update on a TensorDict batch."""
        values = batch.to(self.device)
        observation = values.get(self.observation_key).float()
        next_observation = values.get(
            ("next",) + _as_tuple(self.observation_key)
        ).float()
        actions = values.get(self.action_key).long().reshape(-1, 1)
        rewards = values.get("reward").float().reshape(-1, 1)
        terminated = values.get("terminated").float().reshape(-1, 1)
        q_values = self.network(observation)
        chosen_q = q_values.gather(1, actions)
        with torch.no_grad():
            next_q = self.target_network(next_observation)
            if (
                self.next_action_mask_key is not None
                and self.next_action_mask_key in values.keys(True, True)
            ):
                next_q = _apply_mask(
                    next_q, values.get(self.next_action_mask_key).bool()
                )
            target = (
                rewards
                + self.config.gamma
                * (1.0 - terminated)
                * next_q.max(dim=-1, keepdim=True).values
            )
        loss = functional.mse_loss(chosen_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        return {
            "gradient_updates": 1.0,
            "dqn/loss": float(loss.detach().cpu()),
            "dqn/q_mean": float(chosen_q.detach().mean().cpu()),
        }


def _as_tuple(key: ObservationKey) -> Tuple[str, ...]:
    return (key,) if isinstance(key, str) else key


def _mask_from_observation(
    observation: TensorDictBase, key: Optional[str], action_count: int
) -> torch.Tensor:
    """Return a boolean action mask, defaulting to all valid actions."""
    if key is None or key not in observation.keys():
        return torch.ones(
            action_count,
            dtype=torch.bool,
            device=next(iter(observation.values())).device,
        )
    mask = observation.get(key).bool()
    if mask.numel() != action_count:
        raise ValueError("action mask length does not match action_count")
    if not bool(mask.any()):
        raise ValueError("an action mask must leave at least one action valid")
    return mask


def _apply_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set invalid action values to negative infinity for greedy selection/max targets."""
    if values.shape[-1] != mask.shape[-1]:
        raise ValueError("action value and mask dimensions do not match")
    return values.masked_fill(~mask, float("-inf"))
