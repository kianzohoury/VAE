
from typing import Dict, Tuple

import torch.nn as nn
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from . import vae


def init_model(model_type: str, **kwargs) -> nn.Module:
    """Initializes model."""
    model_constructor = getattr(vae, model_type)
    model = model_constructor(**kwargs)
    return model


def load_dataset(
    name: str = "mnist",
    splits: Tuple[str] = ("train", "test", "val"),
    val_size: float = 0.1
) -> Dict[str, Dataset]:
    """Loads image tensors from MNIST or CIFAR10 dataset."""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    dataset = getattr(torchvision.datasets, name.upper())
    split_data = {}
    if "train" in splits:
        split_data["train"] = dataset(
            root=f"./{name}",
            download=True,
            transform=transform,
            train=True
        )
    if "test" in splits:
        split_data["test"] = dataset(
            root=f"./{name}",
            download=True,
            transform=transform,
            train=False
        )

    if "val" in splits:
        # create validation split
        indices = list(range(len(split_data["train"])))
        train_indices, val_indices = train_test_split(
            indices, test_size=val_size, random_state=0
        )
        split_data["val"] = Subset(split_data["train"], val_indices)
        split_data["train"] = Subset(split_data["train"], train_indices)
    return split_data
