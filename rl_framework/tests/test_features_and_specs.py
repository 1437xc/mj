"""Tests that FDL rules become independent fixed TensorDict feature fields."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from rlframework.data import FieldSpec, TensorDictSpec
from rlframework.feature import ArrayFeatureAdapter, CallableFeatureAdapter
from rlframework.utils.fdl_bridge import FDLFieldBinding, compile_fdl, load_fdl


class FeatureAdapterTests(unittest.TestCase):
    def test_multiple_fdl_rules_become_separate_tensor_dict_fields(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proto = root / "frame.proto"
            feature = root / "feature.fdl"
            proto.write_text(
                "message Frame { required float hp = 1; required int32 mask = 2; }",
                encoding="utf-8",
            )
            feature.write_text(
                "observation(Frame) -> [ hp: norm(0, 100) ]\n"
                "action_mask(Frame) -> [ mask: one_hot(0, 1) ]\n",
                encoding="utf-8",
            )
            compile_fdl([proto], [feature], root / "generated")
            adapter = load_fdl(
                root / "generated",
                [
                    FDLFieldBinding("obs", feature, "observation"),
                    FDLFieldBinding("action_mask", feature, "action_mask"),
                ],
            )
            encoded = adapter.encode(SimpleNamespace(hp=25.0, mask=1))
            self.assertEqual(encoded.get("obs").tolist(), [0.25])
            self.assertEqual(encoded.get("action_mask").tolist(), [0.0, 1.0])
            self.assertTrue(adapter.spec.schema_id.startswith("fdl:"))
            manual = CallableFeatureAdapter(
                {
                    "obs": lambda frame: [frame.hp / 100.0],
                    "action_mask": lambda frame: [
                        float(frame.mask == 0),
                        float(frame.mask == 1),
                    ],
                },
                adapter.spec,
            )
            reference = manual.encode(SimpleNamespace(hp=25.0, mask=1))
            for key in adapter.spec.fields:
                self.assertTrue(torch.equal(encoded.get(key), reference.get(key)))

    def test_manual_and_array_paths_have_identical_nested_fields(self):
        spec = TensorDictSpec(
            {
                ("unit", "hp"): FieldSpec((1,), torch.float32),
                "mask": FieldSpec((2,), torch.bool),
            }
        )
        manual = CallableFeatureAdapter(
            {
                ("unit", "hp"): lambda frame: [frame.hp / 100],
                "mask": lambda frame: [True, False],
            },
            spec,
        )
        actual = manual.encode(SimpleNamespace(hp=25))
        expected = ArrayFeatureAdapter(spec).encode(
            {"unit": {"hp": torch.tensor([0.25])}, "mask": torch.tensor([True, False])}
        )
        self.assertTrue(torch.equal(actual["unit", "hp"], expected["unit", "hp"]))
        self.assertEqual(actual["mask"].dtype, torch.bool)
        with self.assertRaises(ValueError):
            ArrayFeatureAdapter(spec).encode(
                {
                    "unit": {"hp": torch.tensor([0.25, 0.5])},
                    "mask": torch.tensor([True, False]),
                }
            )

    def test_loader_rejects_missing_and_modified_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proto, feature = root / "f.proto", root / "f.fdl"
            binding = FDLFieldBinding("obs", feature, "main")
            with self.assertRaises(FileNotFoundError):
                load_fdl(root, [binding])
            proto.write_text(
                "message Frame { required float hp = 1; }", encoding="utf-8"
            )
            feature.write_text("main(Frame) -> [ hp ]", encoding="utf-8")
            compile_fdl([proto], [feature], root)
            (root / "f.fdl.py").write_text(
                "raise RuntimeError('must not run')", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_fdl(root, [binding])
