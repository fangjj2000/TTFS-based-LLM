import os, sys

import torch
import torch.nn as nn
import safetensors


def load_weights(path: str, model: nn.Module) -> None:
    file = os.path.join(path, "model.safetensors")
    state = {}
    with safetensors.safe_open(file, framework="pt", device="cuda") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    model.load_state_dict(state)
