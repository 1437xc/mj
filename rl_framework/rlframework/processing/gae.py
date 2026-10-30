"""Return processing is applied to each agent segment before any batching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict.tensordict import TensorDictBase

from rlframework.policy import ActorCriticPolicy


@dataclass(frozen=True)
class GAEProcessor:
    """Compute GAE with explicit bootstrap and trace discounts.

    Collector-provided discounts define the decision clock. Gamma/lambda are
    fallbacks for externally supplied trajectories without discount fields.
    Natural terminals bootstrap zero; time limits use the true final state.
    Recursion always stops at the end of the supplied agent segment.
    """

    gamma: float = 0.99
    gae_lambda: float = 0.95

    def process(
        self, trajectory: TensorDictBase, policy: ActorCriticPolicy
    ) -> TensorDictBase:
        if len(trajectory.batch_size) != 1 or trajectory.batch_size[0] == 0:
            raise ValueError("GAE requires a nonempty one-dimensional agent trajectory")
        values = trajectory.get(("policy", "extras", "value")).detach().cpu()
        rewards = trajectory.get("reward").float().cpu()
        terminated, truncated = (
            trajectory.get("terminated").cpu(),
            trajectory.get("truncated").cpu(),
        )
        discount = trajectory.get("discount", self.gamma * (~terminated).float()).cpu()
        trace = trajectory.get(
            "trace_discount",
            self.gamma * self.gae_lambda * (~(terminated | truncated)).float(),
        ).cpu()
        # Terminal observations may have no legal actions; the critic-only path
        # must not construct an action distribution. RNN bootstrap uses state_out.
        next_values = torch.zeros_like(rewards)
        bootstrap_indices = (discount != 0).nonzero(as_tuple=False).flatten()
        if len(bootstrap_indices):
            selected = trajectory[bootstrap_indices.to(trajectory.device or "cpu")]
            with torch.no_grad():
                predictions = (
                    policy.values(
                        selected.get(("next", "obs")),
                        state=selected.get("state_out", None),
                    )
                    .detach()
                    .cpu()
                )
            if predictions.shape != bootstrap_indices.shape:
                raise ValueError("bootstrap values must match selected observations")
            next_values[bootstrap_indices] = predictions
        if values.shape != rewards.shape or next_values.shape != rewards.shape:
            raise ValueError("critic values must match the one-dimensional time axis")
        advantages, carry = torch.zeros_like(rewards), torch.zeros(
            (), dtype=rewards.dtype
        )
        starts = trajectory.get("episode_start", torch.zeros_like(terminated)).cpu()
        for index in range(len(rewards) - 1, -1, -1):
            if index + 1 < len(rewards) and starts[index + 1]:
                carry = torch.zeros_like(carry)
            delta = (
                rewards[index] + discount[index] * next_values[index] - values[index]
            )
            carry = delta + trace[index] * carry
            advantages[index] = carry
        result = trajectory.clone()
        result.set("advantage", advantages.to(trajectory.device or "cpu"))
        result.set("return", (advantages + values).to(trajectory.device or "cpu"))
        return result
