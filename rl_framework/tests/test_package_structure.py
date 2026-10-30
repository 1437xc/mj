"""Check the public contract/implementation boundary of the package layout."""

import unittest
from pathlib import Path


class PackageStructureTests(unittest.TestCase):
    def test_canonical_components_are_importable(self):
        from rlframework.algorithm.dqn import DQNConfig
        from rlframework.collectors.synchronous import SynchronousStepCollector
        from rlframework.data import TensorDictSpec
        from rlframework.environment import MultiEnv
        from rlframework.feature import FeatureAdapter
        from rlframework.policy import ActorCriticPolicy, Policy
        from rlframework.processing.gae import GAEProcessor
        from rlframework.replay_store import ReplayStore
        from rlframework.utils.rpc import TensorDictRpcServer
        from rlframework.weight_store import WeightStore
        from rlframework.workflow.off_policy import LocalOffPolicyWorkflow

        for value in (
            DQNConfig,
            SynchronousStepCollector,
            TensorDictSpec,
            MultiEnv,
            FeatureAdapter,
            ActorCriticPolicy,
            Policy,
            GAEProcessor,
            ReplayStore,
            TensorDictRpcServer,
            WeightStore,
            LocalOffPolicyWorkflow,
        ):
            self.assertIsNotNone(value)

    def test_contracts_and_implementations_have_separate_locations(self):
        package = Path(__file__).resolve().parents[1] / "rlframework"
        for name in (
            "algorithm", "collectors", "environments", "processing",
            "replay", "weight", "workflow", "utils",
        ):
            self.assertTrue((package / name / "__init__.py").is_file(), name)
        for name in (
            "collector",
            "data",
            "environment",
            "feature",
            "policy",
            "learner",
            "replay_store",
            "weight_store",
        ):
            self.assertTrue((package / (name + ".py")).is_file(), name)
            self.assertFalse((package / name).exists(), name)

        for old in (
            "sequence.py", "processor.py", "parallel.py", "runtime.py",
            "recurrent.py", "named_policy.py", "distribution.py", "fdl_bridge.py",
            "replay/core.py", "weight/core.py", "workflow/local.py", "features",
        ):
            self.assertFalse((package / old).exists(), old)

    def test_root_exports_contracts_and_data_only(self):
        import rlframework
        from rlframework.data import SequenceBatch
        from rlframework.feature import ArrayFeatureAdapter, CallableFeatureAdapter
        from rlframework.replay_store import ReplayStore
        from rlframework.weight_store import WeightStore

        self.assertIs(rlframework.SequenceBatch, SequenceBatch)
        self.assertIs(rlframework.ReplayStore, ReplayStore)
        self.assertIs(rlframework.WeightStore, WeightStore)
        self.assertEqual(ArrayFeatureAdapter.__module__, "rlframework.feature")
        self.assertEqual(CallableFeatureAdapter.__module__, "rlframework.feature")
        for name in (
            "ParallelEpisodeCollector", "InMemoryReplayBuffer", "GAEProcessor",
            "RecurrentCategoricalPolicy", "ArrayFeatureAdapter", "TensorDictRpcServer",
        ):
            self.assertNotIn(name, rlframework.__all__)
            self.assertFalse(hasattr(rlframework, name))
