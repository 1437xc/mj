"""Export Python 3.6 source ZIP and separate CPU state-dict deployment weights."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import torch
from .model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu")
    cfg = state["config"]
    model = Model(cfg["hidden"])
    weights = {
        k[len("module.") :]: v.cpu()
        for k, v in state["policy"].items()
        if k.startswith("module.")
    }
    model.load_state_dict(weights, strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    game = __package__
    weight_name = game + "_model.pt"
    torch.save(
        weights,
        args.output / weight_name,
        _use_new_zipfile_serialization=False,
        pickle_protocol=2,
    )
    source = Path(__file__).parent
    excluded = {"env.py", "train.py", "policy.py", "evaluate.py", "export_bot.py"}
    with zipfile.ZipFile(
        args.output / (game + "_bot.zip"), "w", zipfile.ZIP_DEFLATED
    ) as z:
        z.writestr(
            "__main__.py", "from " + game + ".botzone.runner import main\nmain()\n"
        )
        z.writestr(
            game + "/bot_config.json",
            json.dumps(
                {
                    "hidden": cfg["hidden"],
                    "capacity": cfg["capacity"],
                    "weight_path": "data/" + weight_name,
                }
            ),
        )
        for p in source.rglob("*.py"):
            rel = p.relative_to(source)
            if p.name in excluded or any(
                x in rel.parts for x in ("tests", "__pycache__")
            ):
                continue
            z.write(p, game + "/" + rel.as_posix())
    manifest = {
        "game": game,
        "training_counters": state["counters"],
        "iteration": state["iteration"],
        "config": cfg,
        "source_checkpoint_sha256": hashlib.sha256(
            args.checkpoint.read_bytes()
        ).hexdigest(),
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.output.iterdir()
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
