"""Algorithm-specific policies, learners and trajectory processors."""

from .distributions import ConditionalCategorical, masked_categorical
from .named_policy import NamedCategoricalPolicy
from .recurrent import GRUActorCritic, RecurrentCategoricalPolicy
from rlframework.algorithm.a2c import A2CConfig, A2CLearner
from rlframework.algorithm.a3c import A3CSharedLearner
from rlframework.algorithm.ddpg import (
    DDPGActor,
    DDPGConfig,
    DDPGCritic,
    DDPGLearner,
    DDPGPolicy,
)
from rlframework.algorithm.dqn import DQNConfig, DQNLearner, DQNPolicy, MLPQNetwork
from rlframework.algorithm.pg import (
    MLPCategoricalPolicy,
    MonteCarloReturnProcessor,
    PGConfig,
    PGLearner,
    PGPolicy,
)
from rlframework.algorithm.ppo import MLPActorCritic, PPOConfig, PPOLearner, PPOPolicy
from rlframework.algorithm.sac import SACActor, SACConfig, SACLearner, SACPolicy


__all__ = [
    "ConditionalCategorical",
    "masked_categorical",
    "NamedCategoricalPolicy",
    "GRUActorCritic",
    "RecurrentCategoricalPolicy",
    "A2CConfig",
    "A2CLearner",
    "A3CSharedLearner",
    "DDPGActor",
    "DDPGConfig",
    "DDPGCritic",
    "DDPGLearner",
    "DDPGPolicy",
    "DQNConfig",
    "DQNLearner",
    "DQNPolicy",
    "MLPActorCritic",
    "MLPCategoricalPolicy",
    "MLPQNetwork",
    "MonteCarloReturnProcessor",
    "PGConfig",
    "PGLearner",
    "PGPolicy",
    "PPOConfig",
    "PPOLearner",
    "PPOPolicy",
    "SACActor",
    "SACConfig",
    "SACLearner",
    "SACPolicy",
]
