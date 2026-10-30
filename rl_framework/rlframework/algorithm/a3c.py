"""Thread-safe A3C-style shared learner with explicit local-policy synchronization."""

from __future__ import annotations

import threading
from typing import Mapping, Union

import torch
from tensordict.tensordict import TensorDictBase

from rlframework.algorithm.a2c import A2CConfig, a2c_objective
from rlframework.policy import ActorCriticPolicy


class A3CSharedLearner:
    """A small in-process parameter server for independently collecting A3C workers.

    Each worker owns a local :class:`ActorCriticPolicy` copy, computes its
    gradients on a rollout, and calls :meth:`update_from`.  Parameter application
    is locked, after which the worker copy is refreshed from the global policy.
    The class intentionally does not prescribe actor scheduling or transport.
    """

    def __init__(
        self,
        global_policy: ActorCriticPolicy,
        config: A2CConfig = A2CConfig(),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.global_policy = global_policy.to(device)
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            self.global_policy.parameters(), lr=config.learning_rate
        )
        self._lock = threading.Lock()
        self._version = 0

    @property
    def version(self) -> int:
        """Return the latest globally applied gradient version."""
        with self._lock:
            return self._version

    def sync_to(self, local_policy: ActorCriticPolicy) -> int:
        """Copy the current global parameters into one worker-local policy."""
        with self._lock:
            local_policy.load_state_dict(self.global_policy.state_dict())
            return self._version

    def update_from(
        self, local_policy: ActorCriticPolicy, batch: TensorDictBase
    ) -> Mapping[str, float]:
        """Apply gradients computed from a local rollout, then refresh that worker.

        The forward/backward pass happens on the worker policy before obtaining
        the parameter-server lock.  Only assigning gradients and updating the
        shared optimizer are serialized, matching the essential A3C topology.
        """
        local_values = batch.to(self.device)
        local_policy.to(self.device)
        local_policy.zero_grad()
        loss, metrics = a2c_objective(local_policy, local_values, self.config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            local_policy.parameters(), self.config.max_grad_norm
        )
        with self._lock:
            self.optimizer.zero_grad()
            for global_parameter, local_parameter in zip(
                self.global_policy.parameters(), local_policy.parameters()
            ):
                if local_parameter.grad is not None:
                    global_parameter.grad = local_parameter.grad.detach().clone()
            self.optimizer.step()
            self._version += 1
            local_policy.load_state_dict(self.global_policy.state_dict())
            version = self._version
        result = {
            name.replace("a2c/", "a3c/"): float(value.detach().cpu())
            for name, value in metrics.items()
        }
        result["a3c/version"] = float(version)
        return result
