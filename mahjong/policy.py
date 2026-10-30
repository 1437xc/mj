"""Training-only policy and actor factory; model.py stays deployment-compatible."""

from functools import partial
import torch
from rlframework.algorithm.named_policy import NamedCategoricalPolicy
from rlframework.collectors.sparse import SparseDecisionCollector
from rlframework.collectors.routing import StaticPolicyRouter
from rlframework.collector import PolicyRef
from rlframework.collectors.rewards import SharedTensorRewardResolver
from .model import Model
from .env import Environment


def make_policy(hidden=64):
    return NamedCategoricalPolicy(Model(hidden), input_keys=("obs",))


def make_collector(
    worker_id, hidden=64, capacity=128, gamma=0.99, gae_lambda=0.95, seed=0
):
    torch.manual_seed(seed + worker_id)
    policy = make_policy(hidden)
    router = StaticPolicyRouter({str(i): PolicyRef("shared", policy) for i in range(4)})
    return SparseDecisionCollector(
        Environment(capacity),
        router,
        SharedTensorRewardResolver("rewards", {str(i): i for i in range(4)}),
        gamma=gamma,
        gae_lambda=gae_lambda,
        env_id="actor_%d" % worker_id,
    )
