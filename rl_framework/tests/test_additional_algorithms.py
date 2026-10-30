"""Focused update-path tests for the additional discrete and continuous algorithms."""

import copy
import unittest

import torch
from tensordict import TensorDict

from rlframework.algorithm.a2c import A2CConfig, A2CLearner
from rlframework.algorithm.a3c import A3CSharedLearner
from rlframework.algorithm.ddpg import DDPGActor, DDPGConfig, DDPGCritic, DDPGLearner
from rlframework.algorithm.pg import MLPCategoricalPolicy, MonteCarloReturnProcessor, PGPolicy, PGLearner
from rlframework.algorithm.ppo import MLPActorCritic, PPOPolicy
from rlframework.algorithm.sac import SACActor, SACConfig, SACLearner


def discrete_batch():
    """Build one complete discrete trajectory with policy extras for actor-critic tests."""
    count = 5
    return TensorDict(
        {
            "obs": {
                "obs": torch.randn(count, 3),
                "action_mask": torch.ones(count, 2, dtype=torch.bool),
            },
            "action": {"action": torch.randint(0, 2, (count,))},
            "reward": torch.ones(count),
            "terminated": torch.tensor([False, False, False, False, True]),
            "truncated": torch.zeros(count, dtype=torch.bool),
            "next": {
                "obs": {
                    "obs": torch.randn(count, 3),
                    "action_mask": torch.ones(count, 2, dtype=torch.bool),
                }
            },
            "policy": {
                "extras": {"logp": torch.zeros(count), "value": torch.zeros(count)}
            },
        },
        batch_size=[count],
    )


def continuous_batch():
    """Build a normalized continuous-action off-policy batch."""
    count = 6
    return TensorDict(
        {
            "obs": {"obs": torch.randn(count, 3)},
            "action": {"action": torch.randn(count, 2).clamp(-1.0, 1.0)},
            "reward": torch.randn(count),
            "terminated": torch.zeros(count, dtype=torch.bool),
            "truncated": torch.zeros(count, dtype=torch.bool),
            "next": {"obs": {"obs": torch.randn(count, 3)}},
        },
        batch_size=[count],
    )


class AdditionalAlgorithmTests(unittest.TestCase):
    def test_pg_return_processor_and_update(self):
        batch = MonteCarloReturnProcessor().process(discrete_batch())
        learner = PGLearner(PGPolicy(MLPCategoricalPolicy(3, 2)))
        metrics = learner.update(batch)
        self.assertIn("pg/loss", metrics)

    def test_a2c_and_a3c_updates_share_actor_critic_contract(self):
        from rlframework.processing.gae import GAEProcessor

        policy = PPOPolicy(MLPActorCritic(3, 2))
        batch = GAEProcessor().process(discrete_batch(), policy)
        self.assertIn("a2c/loss", A2CLearner(policy, A2CConfig()).update(batch))
        global_policy = PPOPolicy(MLPActorCritic(3, 2))
        local_policy = PPOPolicy(copy.deepcopy(global_policy.module))
        shared = A3CSharedLearner(global_policy)
        shared.sync_to(local_policy)
        metrics = shared.update_from(local_policy, batch)
        self.assertEqual(metrics["a3c/version"], 1.0)

    def test_ddpg_and_sac_updates_on_normalized_continuous_actions(self):
        batch = continuous_batch()
        ddpg = DDPGLearner(DDPGActor(3, 2), DDPGCritic(3, 2), DDPGConfig())
        self.assertIn("ddpg/actor_loss", ddpg.update(batch))
        sac = SACLearner(
            SACActor(3, 2), DDPGCritic(3, 2), DDPGCritic(3, 2), SACConfig()
        )
        self.assertIn("sac/actor_loss", sac.update(batch))
