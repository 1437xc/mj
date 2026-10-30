"""Whole-episode actor barriers preserve versions, identity and failure cleanup."""

import os
import unittest
import torch
from tensordict import TensorDict
from rlframework.environment import MultiEnv, MultiEnvStep, AgentStep
from rlframework.collectors.sparse import SparseDecisionCollector
from rlframework.collectors.routing import StaticPolicyRouter
from rlframework.collector import PolicyRef
from rlframework.collectors.parallel import ParallelEpisodeCollector
from rlframework.algorithm.named_policy import NamedCategoricalPolicy


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 3)

    def forward(self, obs):
        x = self.layer(obs)
        return x[..., :2], x[..., 2]


class TinyEnv(MultiEnv):
    def reset(self, config=None):
        self.seed = config["seed"]
        self.t = 0
        return self.result()

    def result(self):
        obs = TensorDict(
            {
                "obs": torch.tensor([float(self.seed)]),
                "action_mask": torch.tensor([True, False]),
            },
            [],
        )
        return MultiEnvStep(
            {"0": AgentStep(obs, reward=1.0, terminated=self.t == 2)}, self.t == 2
        )

    def step(self, actions):
        self.t += 1
        return self.result()


def worker_factory(index):
    policy = NamedCategoricalPolicy(TinyModel())
    router = StaticPolicyRouter({"0": PolicyRef("p", policy)})
    return SparseDecisionCollector(TinyEnv(), router, env_id="worker_%d" % index)


def broken_factory(index):
    raise ValueError("deliberate worker failure")


class ParallelTest(unittest.TestCase):
    def test_versions_seeds_and_whole_episodes(self):
        policy = NamedCategoricalPolicy(TinyModel())
        router = StaticPolicyRouter({"0": PolicyRef("p", policy, version=7)})
        with ParallelEpisodeCollector(
            worker_factory, router, actors=2, episodes_per_actor=2, seed=20, timeout=30
        ) as collector:
            batch = collector.collect_episode(deterministic=True)
            self.assertEqual(
                (batch.env_steps, batch.agent_steps, batch.completed_episodes),
                (8, 8, 4),
            )
            self.assertEqual(
                {int(t.data["obs", "obs"][0, 0]) for t in batch.trajectories},
                {20, 21, 22, 23},
            )
            self.assertEqual({t.policy_version for t in batch.trajectories}, {7})
            self.assertEqual(len(set(collector.actor_pids)), 2)
            self.assertNotIn(os.getpid(), collector.actor_pids)
            router.set_version("p", 8)
            next_batch = collector.collect_episode()
            self.assertEqual({t.policy_version for t in next_batch.trajectories}, {8})
            self.assertEqual(collector.episode_cursor, 8)
            for t in next_batch.trajectories:
                self.assertEqual(int(t.data["action", "action"].sum()), 0)
                self.assertTrue(t.data["terminated"][-1])
        self.assertEqual(collector.actor_pids, [])

    def test_actor_failure_is_reported_and_reaped(self):
        policy = NamedCategoricalPolicy(TinyModel())
        collector = ParallelEpisodeCollector(
            broken_factory,
            StaticPolicyRouter({"0": PolicyRef("p", policy)}),
            actors=1,
            timeout=30,
        )
        with self.assertRaisesRegex(RuntimeError, "deliberate worker failure"):
            collector.collect_episode()
        self.assertEqual(collector.actor_pids, [])

    def test_timeout_reaps_owned_workers(self):
        policy = NamedCategoricalPolicy(TinyModel())
        collector = ParallelEpisodeCollector(
            worker_factory,
            StaticPolicyRouter({"0": PolicyRef("p", policy)}),
            actors=1,
            timeout=0.00001,
        )
        with self.assertRaises(TimeoutError):
            collector.collect_episode()
        self.assertEqual(collector.actor_pids, [])

    def test_named_policy_probability_and_value(self):
        policy = NamedCategoricalPolicy(TinyModel())
        obs = TensorDict(
            {"obs": torch.ones(3, 1), "action_mask": torch.tensor([[True, False]] * 3)},
            [3],
        )
        actions = TensorDict({"action": torch.zeros(3, dtype=torch.long)}, [3])
        result = policy.evaluate_actions(obs, actions)
        torch.testing.assert_close(result.log_prob, torch.zeros(3))
        self.assertEqual(result.value.shape, torch.Size([3]))
        obs["action_mask"] = torch.zeros(3, 2, dtype=torch.bool)
        self.assertEqual(policy.values(obs).shape, torch.Size([3]))
        with self.assertRaises(ValueError):
            policy.evaluate_actions(obs, actions)


if __name__ == "__main__":
    unittest.main()
