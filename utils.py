
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, Dataset

from . import vae


def init_model(model_type: str, device: str = "cpu", **kwargs) -> nn.Module:
    """Initializes model."""
    filtered_kwargs = {}
    model_constructor = getattr(vae, model_type)
    params = set(inspect.signature(model_constructor).parameters.keys())
    # filter out irrelevant arguments
    for key, val in kwargs.items():
        if key in params:
            filtered_kwargs[key] = val
    model = model_constructor(**filtered_kwargs).to(device)
    return model


def load_from_checkpoint(checkpoint: str, device: str = "cpu") -> nn.Module:
    """Loads weights from a given checkpoint and returns the model."""
    state_dict = torch.load(checkpoint, map_location=device)

    # initialize model
    model_type = Path(checkpoint).stem.split("_")[0]
    model = init_model(model_type, **state_dict["config"]).to(device)
    model.load_state_dict(state_dict["model"])
    return model


def filter_by_digit(dataset: Dataset, digit: int) -> Subset:
    """Returns a subset of the MNIST dataset given a digit class."""
    indices = []
    for idx, sample in enumerate(dataset):
        if sample[-1] == digit:
            indices.append(idx)
    return Subset(dataset, indices)


def load_mnist(root: str = "./mnist", split: str = "train") -> Dataset:
    """Loads MNIST dataset split (and downloads to root if necessary)."""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.MNIST(
        root=root,
        download=True,
        transform=transform,
        train=split == "train"
    )
    return dataset


def train_val_split(
    train_dataset: Dataset, val_size: float = 0.1
) -> Tuple[Subset, Subset]:
    """Partitions MNIST training set into train and validation splits."""
    # create validation split
    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, random_state=0
    )
    val_dataset = Subset(train_dataset, val_indices)
    train_dataset = Subset(train_dataset, train_indices)
    return train_dataset, val_dataset


def load_dataset_splits(
    root: str = "./mnist",
    splits: Union[Tuple, List] = ("train", "test", "val"),
    val_size: float = 0.1
) -> Dict[str, Subset]:
    """Loads image tensors from MNIST dataset."""
    split_data = {}
    for split in ["train", "test"]:
        if split in splits:
            split_data[split] = load_mnist(root, split)

    # create train and validation splits
    if val_size > 0 and "val" in splits:
        train_data, val_data = train_val_split(split_data["train"], val_size)
        split_data["train"], split_data["val"] = train_data, val_data
    return split_data


def create_dataloaders(
    dataset_splits: Dict[str, Subset],
    batch_size: int = 1024,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """Creates dataloaders for each dataset split."""
    dataloaders = {}
    for split, dataset in dataset_splits.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
    return dataloaders
