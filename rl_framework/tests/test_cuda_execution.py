"""CUDA boundary tests for CPU environment data and GPU learners.

The test is skipped on developer machines without CUDA.  It deliberately keeps
the observation TensorDict on CPU: that is the normal output of ``MultiEnv``
implementations and therefore exercises the actor-side device boundary.
"""

import unittest

import torch
from tensordict import TensorDict

from rlframework.algorithm.ddpg import DDPGActor, DDPGPolicy
from rlframework.algorithm.dqn import DQNConfig, DQNLearner, DQNPolicy, MLPQNetwork
from rlframework.algorithm.pg import MLPCategoricalPolicy, PGPolicy
from rlframework.algorithm.ppo import MLPActorCritic, PPOConfig, PPOLearner, PPOPolicy
from rlframework.algorithm.sac import SACActor, SACPolicy


def _discrete_batch(count: int = 4) -> TensorDict:
    """Create a CPU transition batch matching the DQN/PPO learner contract."""
    return TensorDict(
        {
            "obs": {
                "obs": torch.randn(count, 3),
                "action_mask": torch.ones(count, 2, dtype=torch.bool),
            },
            "action": {"action": torch.randint(0, 2, (count,))},
            "reward": torch.randn(count),
            "terminated": torch.zeros(count, dtype=torch.bool),
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


@unittest.skipUnless(
    torch.cuda.is_available(), "CUDA is required for this execution-boundary test"
)
class CudaExecutionTests(unittest.TestCase):
    """Verify actor inference and learner updates across CPU/GPU boundaries."""

    def test_builtin_actor_policies_accept_cpu_observations_on_cuda(self) -> None:
        """Policies moved by their learners still act on a CPU environment observation."""
        observation = TensorDict(
            {"obs": torch.randn(3), "action_mask": torch.tensor([True, True])},
            batch_size=[],
        )
        dqn_network = MLPQNetwork(3, 2)
        DQNLearner(dqn_network, device="cuda:0")
        policies = (
            DQNPolicy(dqn_network, 2),
            PPOPolicy(MLPActorCritic(3, 2)).to("cuda:0"),
            PGPolicy(MLPCategoricalPolicy(3, 2)).to("cuda:0"),
            DDPGPolicy(DDPGActor(3, 1)).to("cuda:0"),
            SACPolicy(SACActor(3, 1)).to("cuda:0"),
        )
        for policy in policies:
            with self.subTest(policy=type(policy).__name__):
                output = policy.act(observation)
                self.assertTrue(output.action.get("action").is_cuda)

    def test_dqn_and_ppo_learners_move_cpu_batches_to_cuda(self) -> None:
        """The two main discrete learners accept collector/replay CPU batches."""
        batch = _discrete_batch()
        dqn = DQNLearner(
            MLPQNetwork(3, 2), DQNConfig(target_update_interval=1), device="cuda:0"
        )
        self.assertIn("dqn/loss", dqn.update(batch))
        policy = PPOPolicy(MLPActorCritic(3, 2))
        ppo = PPOLearner(policy, PPOConfig(epochs=1, minibatch_size=2), device="cuda:0")
        processed = (
            batch.clone().set("advantage", torch.randn(4)).set("return", torch.randn(4))
        )
        self.assertIn("ppo/loss", ppo.update(processed))
