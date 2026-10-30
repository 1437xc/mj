"""Fresh-process checks that import caches cannot hide dependency coupling."""

import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
GUARD = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in BLOCKED):
            raise AssertionError('forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Block())
"""


class DependencyBoundaryTests(unittest.TestCase):
    def run_isolated(self, blocked, source):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            str(ROOT / item) for item in ("rl_framework", "fdl")
        )
        script = "BLOCKED = " + repr(blocked) + "\n" + GUARD + source
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_root_import_does_not_load_concrete_components(self):
        self.run_isolated(
            [
                "fdl", "rlframework.algorithm", "rlframework.collectors",
                "rlframework.environments", "rlframework.processing", "rlframework.replay",
                "rlframework.weight", "rlframework.workflow", "rlframework.utils.rpc",
                "rlframework.utils.fdl_bridge",
            ],
            "import rlframework\nassert rlframework.SequenceBatch.__module__ == 'rlframework.data'\n",
        )

    def test_rl_core_runs_with_fdl_unavailable(self):
        self.run_isolated(
            ["fdl"],
            """
import torch
import rlframework
from rlframework.algorithm import DQNPolicy, DQNLearner, MLPQNetwork
from tensordict import TensorDict
network = MLPQNetwork(4, 2)
policy = DQNPolicy(network, 2)
policy.act(TensorDict({'obs': torch.zeros(4)}, batch_size=[]))
learner = DQNLearner(network)
batch = TensorDict({'obs': {'obs': torch.zeros(2, 4)},
                    'action': {'action': torch.tensor([0, 1])},
                    'reward': torch.ones(2), 'terminated': torch.tensor([False, True]),
                    'truncated': torch.tensor([False, False]),
                    'next': {'obs': {'obs': torch.ones(2, 4)}}}, batch_size=[2])
learner.update(batch)
assert 'rlframework.utils.fdl_bridge' not in sys.modules
""",
        )

    def test_fdl_compiles_and_encodes_without_rl_stack(self):
        self.run_isolated(
            ["rlframework", "torch", "tensordict"],
            """
from fdl import Compiler
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
with TemporaryDirectory() as folder:
    root = Path(folder)
    proto, feature = root/'frame.proto', root/'feature.fdl'
    proto.write_text('message Frame { required float hp = 1; }')
    feature.write_text('main(Frame) -> [ hp: norm(0,100) ]')
    result = Compiler().compile_paths([proto], [feature]).require_success()
    namespace = {}
    exec(next(iter(result.python_sources.values())), namespace)
    assert namespace['main'](SimpleNamespace(hp=25)) == [.25]
""",
        )

    def test_generated_runtime_does_not_import_parser(self):
        self.run_isolated(
            ["pyparsing", "fdl.project", "fdl.semantic", "fdl.proto_parser"],
            """
from pathlib import Path
from types import SimpleNamespace
from rlframework.utils.fdl_bridge import load_fdl, FDLFieldBinding
root = Path("rl_examples/03_fdl_treasure_grid/generated")
assert root.joinpath('features.json').is_file(), 'build example features before this test'
adapter = load_fdl(root, [FDLFieldBinding('obs', Path('feature.fdl'), 'observation')])
assert adapter.spec.fields['obs'].shape == (86,)
""",
        )
