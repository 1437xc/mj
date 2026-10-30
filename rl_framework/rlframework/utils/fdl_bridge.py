"""Optional FDL build/load integration; never imported by the RL core.

compile_fdl is an explicit build step. load_fdl loads generated functions, not
the compiler. Generated functions still use FDL's numerical runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import torch

from rlframework.data import FieldSpec, NestedKey, TensorDictSpec
from rlframework.feature import CallableFeatureAdapter


@dataclass(frozen=True)
class FDLFieldBinding:
    field: NestedKey
    rule_file: Path
    rule_name: str
    arguments: Tuple[Any, ...] = ()
    dtype: torch.dtype = torch.float32


def compile_fdl(
    proto_paths: Sequence[Path],
    fdl_paths: Sequence[Path],
    output_directory: Path,
    operator_paths: Sequence[Path] = (),
) -> Path:
    """Write generated modules, docs and a relative, content-addressed manifest."""
    from fdl.project import Compiler

    names = [Path(path).name for path in fdl_paths]
    if len(set(names)) != len(names):
        raise ValueError("FDL source basenames must be unique within an artifact")
    result = (
        Compiler()
        .compile_paths(proto_paths, fdl_paths, operator_paths)
        .require_success()
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rules = {}
    for source, generated in result.python_sources.items():
        filename = Path(source).name + ".py"
        (output_directory / filename).write_bytes(generated.encode("utf-8"))
        (output_directory / (Path(source).name + ".doc")).write_text(
            result.documentation_sources[source], encoding="utf-8"
        )
        lengths = {
            item.name: result.analysis.rule_lengths[id(item.rule)]
            for item in result.analysis.functions.values()
            if hasattr(item, "rule")
            and Path(item.path).resolve() == Path(source).resolve()
        }
        rules[Path(source).name] = {
            "module": filename,
            "sha256": hashlib.sha256(generated.encode("utf-8")).hexdigest(),
            "lengths": lengths,
        }
    target = output_directory / "features.json"
    target.write_text(json.dumps(rules, indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_fdl(
    directory: Path, bindings: Sequence[FDLFieldBinding]
) -> CallableFeatureAdapter:
    """Load and verify generated files; never silently invoke the compiler."""
    directory = Path(directory)
    path = directory / "features.json"
    if not path.is_file():
        raise FileNotFoundError(
            "compile FDL features before training: missing " + str(path)
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not bindings or len({binding.field for binding in bindings}) != len(bindings):
        raise ValueError("feature bindings must be nonempty and have unique fields")
    modules, encoders, fields = {}, {}, {}
    for binding in bindings:
        entry = manifest[Path(binding.rule_file).name]
        filename = entry["module"]
        if Path(filename).name != filename:
            raise ValueError(
                "generated module must be directly inside the artifact directory"
            )
        generated_path = directory / filename
        payload = generated_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError("generated feature module does not match its manifest")
        if filename not in modules:
            module_spec = importlib.util.spec_from_file_location(
                "_fdl_" + digest, str(generated_path)
            )
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            modules[filename] = module
        function = getattr(modules[filename], binding.rule_name)
        encoders[binding.field] = _bind(function, binding.arguments)
        fields[binding.field] = FieldSpec(
            (entry["lengths"][binding.rule_name],), binding.dtype
        )
    identity = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return CallableFeatureAdapter(encoders, TensorDictSpec(fields, "fdl:" + identity))


def _bind(function, arguments):
    return lambda observation: function(observation, *arguments)
