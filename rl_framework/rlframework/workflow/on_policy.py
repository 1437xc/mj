"""Joint collection followed by per-policy processing, learning and publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from rlframework.data import Collection
from rlframework.learner import BatchLearner
from rlframework.policy import ActorCriticPolicy
from rlframework.processing.gae import GAEProcessor
from rlframework.processing.sequence import SequenceBatcher
from rlframework.utils.module import module_parameters
from rlframework.weight.memory import InMemoryWeightStore


@dataclass
class OnPolicyStack:
    """One policy's return processor, optimizer owner and optional sequence batcher."""

    policy: ActorCriticPolicy
    processor: GAEProcessor
    learner: BatchLearner
    batcher: Optional[SequenceBatcher] = None


class MultiPolicyOnPolicyWorkflow:
    """Collect jointly, process agents independently, update each policy once.

    Parameter sharing uses one policy id and one stack. Different stacks may
    not share parameter objects, because two optimizers would silently update
    the same weights. Published versions tag subsequent collected actions.
    """

    def __init__(
        self, collector, stacks: Mapping[str, OnPolicyStack], weight_store=None
    ):
        if not stacks:
            raise ValueError("at least one on-policy stack is required")
        self.collector, self.stacks = collector, dict(stacks)
        self.weight_store = (
            weight_store if weight_store is not None else InMemoryWeightStore()
        )
        self.last_collection = None
        routed = collector.router.policies
        if set(routed) != set(self.stacks):
            raise ValueError("workflow stacks must cover exactly the routed policy ids")
        owned = set()
        for policy_id, stack in self.stacks.items():
            parameters = {id(p) for p in stack.policy.parameters()}
            if owned & parameters:
                raise ValueError(
                    "different policy stacks cannot share parameters; use one shared stack"
                )
            owned.update(parameters)
            if stack.learner.policy is not stack.policy:
                raise ValueError("stack learner must own the routed policy")
            if routed[policy_id] is not stack.policy:
                raise ValueError(
                    "routed policy must be the policy owned by its learner stack"
                )
            if (
                collector.gamma != stack.processor.gamma
                or collector.gae_lambda != stack.processor.gae_lambda
            ):
                raise ValueError(
                    "collector and return processor must agree on discounts"
                )
            self._publish(policy_id, stack)

    def _publish(self, policy_id, stack):
        snapshot = self.weight_store.publish(policy_id, module_parameters(stack.policy))
        self.collector.router.set_version(policy_id, snapshot.version)

    def step(
        self, min_env_steps=None, config=None, episode_configs=None, on_episode=None
    ):
        """Run one iteration; sparse pending actions close before weights change.

        Synchronous rollouts may stop mid-episode. Sparse rollouts keep pending
        decisions in the collector; this synchronous learner finishes that live
        episode before updating, so no decision spans two behavior snapshots.
        """
        if episode_configs is not None:
            if min_env_steps is not None:
                raise ValueError("choose episode_configs or a step budget")
            collections = []
            for episode_config in episode_configs:
                item = self.collector.collect_episode(config=episode_config)
                collections.append(item)
                if on_episode is not None:
                    on_episode(item)
            if not collections:
                raise ValueError("episode_configs must be nonempty")
            collected = Collection(
                [t for c in collections for t in c.trajectories],
                sum(c.env_steps for c in collections),
                sum(c.agent_steps for c in collections),
                sum(c.completed_episodes for c in collections),
            )
        else:
            collected = (
                self.collector.collect_episode(config=config)
                if min_env_steps is None
                else self.collector.collect(min_env_steps, config=config)
            )
        if self.collector.has_pending:
            tail = self.collector.collect_episode()
            collected = Collection(
                collected.trajectories + tail.trajectories,
                collected.env_steps + tail.env_steps,
                collected.agent_steps + tail.agent_steps,
                collected.completed_episodes + tail.completed_episodes,
            )
        self.last_collection = collected
        grouped = {}
        # All bootstrap values are computed before the first parameter update.
        for trajectory in collected.trajectories:
            stack = self.stacks[trajectory.policy_id]
            grouped.setdefault(trajectory.policy_id, []).append(
                stack.processor.process(trajectory.data, stack.policy)
            )
        metrics = {
            "env_steps": float(collected.env_steps),
            "agent_steps": float(collected.agent_steps),
            "completed_episodes": float(collected.completed_episodes),
            "gradient_updates": 0.0,
        }
        for policy_id, trajectories in grouped.items():
            stack = self.stacks[policy_id]
            batch = (
                stack.batcher.batch(trajectories)
                if stack.batcher is not None
                else torch.cat(trajectories, 0)
            )
            result = stack.learner.update(batch)
            metrics["gradient_updates"] += result["gradient_updates"]
            metrics[policy_id + "/trajectories"] = float(len(trajectories))
            metrics[policy_id + "/valid_samples"] = float(
                sum(len(t) for t in trajectories)
            )
            metrics.update(
                {policy_id + "/" + name: value for name, value in result.items()}
            )
            self._publish(policy_id, stack)
            metrics[policy_id + "/weight_version"] = float(
                self.weight_store.latest(policy_id).version
            )
        return metrics
