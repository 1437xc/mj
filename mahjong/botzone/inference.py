"""Deployment-only model loader; no TensorDict, framework or training imports."""

import json
import os
import pkgutil
import numpy as np
import torch
from ..model import Model


def load():
    game = __package__.split(".")[0]
    config = json.loads(pkgutil.get_data(game, "bot_config.json").decode("utf-8"))
    torch.set_num_threads(1)
    model = Model(config["hidden"])
    path = os.environ.get("BOT_MODEL_PATH", config["weight_path"])
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    model.eval()

    def choose(arrays):
        with torch.no_grad():
            inputs = [
                torch.from_numpy(np.asarray(arrays[k])).float().unsqueeze(0)
                for k in ("obs",)
            ]
            logits, _ = model(*inputs)
            mask = torch.from_numpy(arrays["action_mask"]).bool().unsqueeze(0)
            if not mask.any().item():
                raise ValueError("No legal action")
            return int(logits.masked_fill(~mask, -1e9).argmax(dim=-1).item())

    return choose, config
