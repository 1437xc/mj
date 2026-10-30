"""Smoke tests for TensorDict transitions, DQN and PPO update paths."""

import unittest

import torch
from tensordict import TensorDict

from rlframework.algorithm.dqn import DQNConfig, DQNLearner, DQNPolicy, MLPQNetwork
from rlframework.algorithm.ppo import MLPActorCritic, PPOConfig, PPOLearner, PPOPolicy
from rlframework.collector import PolicyRef
from rlframework.collectors.rewards import SharedTensorRewardResolver
from rlframework.collectors.sparse import SparseDecisionCollector
from rlframework.collectors.routing import StaticPolicyRouter
from rlframework.collectors.synchronous import SynchronousStepCollector
from rlframework.environment import AgentStep, MultiEnv, MultiEnvStep
from rlframework.processing.gae import GAEProcessor
from rlframework.replay.memory import InMemoryReplayBuffer


class CounterEnv(MultiEnv):
    """Deterministic small environment that exposes an observation and action mask."""

    def __init__(self, length=3):
        self.length = length
        self.index = 0

    def reset(self, config=None):
        self.index = 0
        return MultiEnvStep({"agent": AgentStep(self._observation())}, False)

    def step(self, actions):
        self.index += 1
        terminal = self.index >= self.length
        reward = float(actions["agent"].get("action").item() == 1)
        return MultiEnvStep(
            {
                "agent": AgentStep(
                    self._observation(), reward=reward, terminated=terminal
                )
            },
            terminal,
        )

    def _observation(self):
        return TensorDict(
            {
                "obs": torch.tensor([float(self.index), 1.0]),
                "action_mask": torch.tensor([True, True]),
            },
            batch_size=[],
        )


class AlternatingEnv(MultiEnv):
    """Two agents act on different ticks and receive terminal vector rewards."""

    def __init__(self):
        self.index = 0

    def reset(self, config=None):
        self.index = 0
        return MultiEnvStep({"a": AgentStep(self._observation(0))}, False)

    def step(self, actions):
        self.index += 1
        if self.index == 1:
            self.assert_actions(actions, {"a"})
            return MultiEnvStep({"b": AgentStep(self._observation(10))}, False)
        if self.index == 2:
            self.assert_actions(actions, {"b"})
            return MultiEnvStep({"a": AgentStep(self._observation(1))}, False)
        self.assert_actions(actions, {"a"})
        return MultiEnvStep(
            {},
            True,
            TensorDict(
                {
                    "terminal_rewards": torch.tensor([2.0, 3.0]),
                    "terminated": torch.tensor(True),
                    "truncated": torch.tensor(False),
                },
                batch_size=[],
            ),
        )

    @staticmethod
    def assert_actions(actions, expected):
        if set(actions) != expected:
            raise AssertionError("unexpected active-agent action set")

    @staticmethod
    def _observation(value):
        return TensorDict(
            {
                "obs": torch.tensor([float(value), 1.0]),
                "action_mask": torch.tensor([True, True]),
            },
            batch_size=[],
        )


class CollectionAndAlgorithmTests(unittest.TestCase):
    def test_collector_builds_named_nested_transitions(self):
        policy = DQNPolicy(MLPQNetwork(2, 2), 2, epsilon=0.0)
        collector = SynchronousStepCollector(
            CounterEnv(), StaticPolicyRouter({"agent": PolicyRef("main", policy)})
        )
        records = collector.collect_episode(deterministic=True).records("main")
        self.assertEqual(len(records), 3)
        self.assertEqual(tuple(records[0].get(("obs", "obs")).shape), (2,))
        self.assertIn(("next", "obs", "action_mask"), records[0].keys(True, True))
        self.assertTrue(bool(records[-1].get("terminated")))

    def test_dqn_replay_and_update(self):
        policy = DQNPolicy(MLPQNetwork(2, 2), 2, epsilon=0.0)
        collector = SynchronousStepCollector(
            CounterEnv(length=4),
            StaticPolicyRouter({"agent": PolicyRef("main", policy)}),
        )
        replay = InMemoryReplayBuffer(16, seed=1)
        replay.add(collector.collect_episode(deterministic=True).records("main"))
        metrics = DQNLearner(
            policy.network, DQNConfig(target_update_interval=1)
        ).update(replay.sample(3))
        self.assertIn("dqn/loss", metrics)
        self.assertGreaterEqual(metrics["dqn/loss"], 0.0)

    def test_ppo_gae_and_update(self):
        policy = PPOPolicy(MLPActorCritic(2, 2))
        collector = SynchronousStepCollector(
            CounterEnv(length=5),
            StaticPolicyRouter({"agent": PolicyRef("main", policy)}),
        )
        records = collector.collect_episode().records("main")
        from rlframework.utils.tensor import stack_records

        batch = GAEProcessor().process(stack_records(records), policy)
        self.assertIn("advantage", batch.keys())
        metrics = PPOLearner(policy, PPOConfig(epochs=1, minibatch_size=2)).update(
            batch
        )
        self.assertIn("ppo/loss", metrics)

    def test_sparse_collector_keeps_pending_decisions_per_agent(self):
        policy = PPOPolicy(MLPActorCritic(2, 2))
        collector = SparseDecisionCollector(
            AlternatingEnv(),
            StaticPolicyRouter(
                {
                    "a": PolicyRef("shared", policy),
                    "b": PolicyRef("shared", policy),
                }
            ),
            SharedTensorRewardResolver("terminal_rewards", {"a": 0, "b": 1}),
        )
        trajectories = collector.collect_episode(deterministic=True)
        self.assertEqual(trajectories.env_steps, 3)
        self.assertEqual(trajectories.agent_steps, 3)
        trajectories = {
            t.key.agent_id: list(t.data.unbind(0)) for t in trajectories.trajectories
        }
        self.assertEqual(set(trajectories), {"a", "b"})
        self.assertEqual(len(trajectories["a"]), 2)
        self.assertEqual(len(trajectories["b"]), 1)
        self.assertEqual(float(trajectories["a"][-1].get("reward")), 2.0)
        self.assertEqual(float(trajectories["b"][-1].get("reward")), 3.0)
        self.assertTrue(bool(trajectories["a"][-1].get("terminated")))
        self.assertTrue(bool(trajectories["b"][-1].get("terminated")))
