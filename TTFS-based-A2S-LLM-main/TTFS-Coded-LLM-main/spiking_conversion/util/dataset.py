import json

import torch, torch.utils.data
from torchvision import datasets, transforms
import timm.data


def imagenet_loader(
    path: str,
    weight_dir: str,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    with open(f"{weight_dir}/config.json", "r") as file:
        pretrained_cfg = json.load(file)["pretrained_cfg"]
    config = timm.data.resolve_data_config(pretrained_cfg=pretrained_cfg)
    transforms = timm.data.create_transform(**config, is_training=False)
    print(transforms)
    return torch.utils.data.DataLoader(
        datasets.ImageFolder(path, transforms),
        batch_size=batch_size,
        shuffle=False,
    )
