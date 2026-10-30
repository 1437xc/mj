"""Core field, trajectory, collection and sequence-batch data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Union

import torch
from tensordict.tensordict import TensorDictBase

from rlframework.utils.module import module_device



NestedKey = Union[str, Tuple[str, ...]]


TensorTreeValue = Union[torch.Tensor, Mapping[str, Any]]


@dataclass(frozen=True)
class TrajectoryKey:
    """Identity of one agent's contiguous segment; policy sharing never merges time."""

    env_id: str
    episode_id: int
    agent_id: str
    segment_id: int


@dataclass
class Trajectory:
    """Time-ordered records from one agent and one behavior-policy version."""

    key: TrajectoryKey
    policy_id: str
    policy_version: int
    data: TensorDictBase


@dataclass
class Collection:
    """Collected segments and actual work, including decisions still pending.

    ``env_steps`` counts joint environment calls; ``agent_steps`` counts actions.
    A sparse decision can close in a later collection than the call that acted.
    """

    trajectories: list[Trajectory]
    env_steps: int
    agent_steps: int
    completed_episodes: int

    def records(self, policy_id: str) -> list[TensorDictBase]:
        """Flatten a route for replay only; process temporal returns before merging."""
        return [
            row
            for trajectory in self.trajectories
            if trajectory.policy_id == policy_id
            for row in trajectory.data.unbind(0)
        ]


@dataclass(frozen=True)
class FieldSpec:
    """Shape and dtype of one TensorDict leaf, excluding leading batch axes."""

    shape: Tuple[int, ...]
    dtype: torch.dtype
    semantic: str = "aux"


@dataclass(frozen=True)
class TensorDictSpec:
    """A minimal, explicit contract for required TensorDict fields."""

    fields: Mapping[NestedKey, FieldSpec]
    schema_id: str = "unspecified"

    def validate(self, batch: TensorDictBase, batch_dims: int = 0) -> None:
        """Raise a descriptive error when a required leaf is absent or incompatible."""
        for key, field_spec in self.fields.items():
            if key not in batch.keys(True, True):
                raise KeyError("missing required TensorDict field {0!r}".format(key))
            value = batch.get(key)
            expected_shape = tuple(batch.batch_size[batch_dims:]) + field_spec.shape
            if tuple(value.shape[batch_dims:]) != expected_shape:
                raise ValueError(
                    "field {0!r} has shape {1}, expected batch suffix {2}".format(
                        key, tuple(value.shape), expected_shape
                    )
                )
            if value.dtype != field_spec.dtype:
                raise TypeError(
                    "field {0!r} has dtype {1}, expected {2}".format(
                        key, value.dtype, field_spec.dtype
                    )
                )


@dataclass
class SequenceBatch:
    """Padded training windows plus detached context used to reconstruct RNN state.

    ``data`` has [batch,time] axes. Prefixes are burn-in only, outside every
    loss and advantage-normalization denominator. State names belong to policies.
    """

    data: TensorDictBase
    valid_mask: torch.Tensor
    initial_state: Optional[TensorDictBase]
    prefixes: list[TensorDictBase]

    def select(self, indices):
        """Select complete sequences, never individual shuffled timesteps."""
        ids = (
            indices.detach().cpu().tolist()
            if isinstance(indices, torch.Tensor)
            else list(indices)
        )
        return SequenceBatch(
            self.data[ids],
            self.valid_mask[ids],
            None if self.initial_state is None else self.initial_state[ids],
            [self.prefixes[i] for i in ids],
        )

    def to(self, device):
        """Move training tensors; actor-side burn-in handles its own device boundary."""
        return SequenceBatch(
            self.data.to(device),
            self.valid_mask.to(device),
            None if self.initial_state is None else self.initial_state.to(device),
            self.prefixes,
        )

    def states(self, policy):
        """Recompute burn-in under current weights with no gradient through prefixes."""
        if self.initial_state is None:
            return None
        states = []
        with torch.no_grad():
            for row, prefix in enumerate(self.prefixes):
                state = self.initial_state[row]
                for record in prefix.unbind(0):
                    if bool(record.get("episode_start")):
                        state = None
                    state = policy.act(
                        record.get("obs"), state, deterministic=True
                    ).next_state
                states.append(state.to(module_device(policy)))
        return torch.stack(states, 0).to_tensordict()
