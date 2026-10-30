"""Sequence packing and learner views with padding masks and burn-in."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from rlframework.data import SequenceBatch
from rlframework.utils.tensor import stack_records


@dataclass(frozen=True)
class SequenceBatcher:
    """Pack complete agent trajectories or bounded windows with optional burn-in.

    Each window starts from its recorded incoming state. A burn-in prefix moves
    that starting point earlier, then reconstructs state without prefix gradients.
    Padding repeats a real observation so legal-action masks remain valid.
    """

    unroll_length: Optional[int] = None
    burn_in: int = 0

    def __post_init__(self):
        if self.unroll_length is not None and self.unroll_length <= 0:
            raise ValueError("unroll_length must be positive")
        if self.burn_in < 0:
            raise ValueError("burn_in must be nonnegative")

    def batch(self, trajectories):
        """Pack processed [T] agent trajectories into one masked [B,T] batch."""
        windows, prefixes, states = [], [], []
        recurrent = None
        for data in trajectories:
            if len(data.batch_size) != 1 or not len(data):
                raise ValueError(
                    "sequence batching requires nonempty agent trajectories"
                )
            has_state = "state_in" in data.keys()
            if recurrent is not None and recurrent != has_state:
                raise ValueError("cannot mix recurrent and stateless trajectories")
            recurrent = has_state
            length = self.unroll_length or len(data)
            for start in range(0, len(data), length):
                prefix_start = max(0, start - self.burn_in)
                windows.append(data[start : start + length])
                prefixes.append(data[prefix_start:start])
                if has_state:
                    states.append(data.get("state_in")[prefix_start])
        if not windows:
            raise ValueError("cannot batch empty trajectories")
        maximum = max(len(window) for window in windows)
        padded, masks = [], []
        for window in windows:
            padding = [window[-1].clone() for _ in range(maximum - len(window))]
            padded.append(stack_records(list(window.unbind(0)) + padding))
            masks.append(torch.arange(maximum) < len(window))
        return SequenceBatch(
            torch.stack(padded, 0).to_tensordict(),
            torch.stack(masks),
            torch.stack(states, 0).to_tensordict() if states else None,
            prefixes,
        )


def training_view(batch, policy):
    """Return data, valid loss positions and recurrent evaluation arguments."""
    if isinstance(batch, SequenceBatch):
        data = batch.data
        return (
            data,
            batch.valid_mask,
            {"state": batch.states(policy), "episode_start": data.get("episode_start")},
        )
    if len(batch.batch_size) != 1 or not len(batch):
        raise ValueError("flat training batches must be nonempty and one-dimensional")
    if "state_in" in batch.keys():
        raise ValueError("recurrent trajectories require SequenceBatcher")
    return (
        batch,
        torch.ones(batch.batch_size, dtype=torch.bool, device=batch.device),
        {},
    )
