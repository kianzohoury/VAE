import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision

from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from . import vae

LATENT_SEARCH_SPACE = [2, 5, 10, 20, 50, 100]
NUM_CLASSES = 10  # same for MNIST and CIFAR-10
BATCH_SIZE = 1024
NUM_WORKERS = 4
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def test_model(model: nn.Module, test_loader):
    """Tests model."""
    total_loss = defaultdict(float)
    model.eval()
    for idx, (img, label) in enumerate(test_loader, 1):
        img = img.to(DEVICE)

        if isinstance(model, vae.ConditionalVAE):
            loss = model.training_step(img, label)
        else:
            loss = model.training_step(img)
        # aggregate losses
        for loss_term, loss_val in loss.items():
            total_loss[loss_term] += loss_val.item()

    # average each loss term
    for loss_term, loss_val in total_loss.items():
        total_loss[loss_term] /= len(test_loader)
    return total_loss


def test_across_classes(
    model_dir: str, dataset: str = "mnist", is_test: bool = False
):
    """Tests models separately for each class (e.g. digits in MNIST)."""
    if is_test:
        datasets = load_dataset(
            name=dataset,
            splits=("test"),
            val_size=0
        )
    else:
        # use validation set (necessary for model selection, and to keep
        # test set independent of this process)
        datasets = load_dataset(
            name=dataset,
            splits=("train", "val"),
            val_size=0.1
        )
    test_dataset = datasets["test"] if is_test else datasets["val"]
    test_losses = defaultdict(partial(defaultdict, partial(defaultdict, list)))
    for class_idx in range(NUM_CLASSES):
        indices = []
        for idx, sample in enumerate(test_dataset):
            if sample[-1] == class_idx:
                indices.append(idx)

        # create dataloader
        test_loader = DataLoader(
            dataset=Subset(test_dataset, indices),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True,
            pin_memory=True
        )

        print(f"Starting testing for class {class_idx}...")
        for checkpoint in list(Path(model_dir).rglob("*.pth")):
            state_dict = torch.load(checkpoint, map_location=DEVICE)

            # initialize model and optimizer
            model = init_model(state_dict["config"]["model_type"]).to(DEVICE)
            num_latent = state_dict["config"]["num_latent"]
            model.load_state_dict(state_dict["model"])

            # test
            test_loss = test_model(model, test_loader)
            for loss_term, loss_val in test_loss.items():
                test_losses[class_idx][loss_term][num_latent].append(loss_val)
                print(f"{loss_term}: {round(loss_val, 3)}")

    # save test results
    print("Saving test results...")
    pkl_filename = f"./{model_dir}/class_results_test.pkl" if is_test else \
        f"./{model_dir}/class_results_val.pkl"
    with open(pkl_filename, mode="wb") as f:
        pickle.dump(test_losses, f)
    print("Finished testing.")


def run_training(
    model_type,
    dataset: str = "mnist",
    validate: bool = True,
    num_epochs: int = 20
):
    """Trains given model type for each latent representation size."""
    datasets = load_dataset(
        name=dataset,
        splits=("train", "val") if validate else ("train"),
        val_size=0 if not validate else 0.1)

    # create dataloaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    if validate:
        val_loader = DataLoader(
            datasets["val"],
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
    else:
        val_loader = None

    config = {
        "num_features": 28 * 28 if dataset == "mnist" else 32 * 32,
        "model_type": model_type
    }
    if model_type == "ConditionalVAE":
        config["num_classes"] = NUM_CLASSES

    # keep track of metrics across all models
    train_losses = defaultdict(partial(defaultdict, list))
    val_losses = defaultdict(partial(defaultdict, list))

    # simple hyperparameter search over latent dimensions
    for num_latent in LATENT_SEARCH_SPACE:

        # initialize model and optimizer
        model_config = dict(config)
        model_config["num_latent"] = num_latent
        model = init_model(**model_config).to(DEVICE)
        optim = AdamW(model.parameters(), LR)
        print(f"Starting training for z-dim={num_latent}.")

        model.train()
        for epoch in range(num_epochs):
            total_loss = defaultdict(float)
            with tqdm(train_loader) as tq:
                tq.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
                for idx, (img, label) in enumerate(tq, 1):
                    img = img.to(DEVICE)

                    if model_type == "ConditionalVAE":
                        loss = model.training_step(img, label)
                    else:
                        loss = model.training_step(img)

                    # backprop + update parameters
                    loss["loss"].backward()
                    optim.step()

                    # clear gradient
                    optim.zero_grad()

                    # aggregate losses
                    for loss_term, loss_val in loss.items():
                        total_loss[loss_term] += loss_val.item()

                    # log results
                    tq.set_postfix({
                        key: val / idx for (key, val) in total_loss.items()
                    })

            for loss_term, loss_val in total_loss.items():
                train_losses[loss_term][num_latent].append(loss_val / idx)

            # run validation
            if validate:
                validation_loss = test_model(model, val_loader)
                for loss_term, loss_val in validation_loss.items():
                    val_losses[loss_term][num_latent].append(loss_val)
                    print(f"{loss_term}: {round(loss_val, 3)}")

        print("Saving model...")
        Path(f"./{model_type}").mkdir(exist_ok=True)  # make directory
        torch.save(
            {"model": model.cpu().state_dict(), "config": model_config},
            f=f"{model_type}/{model_type}_latent_{num_latent}.pth",
        )

    # save epoch history
    print("Saving loss history...")
    with open(f"./{model_type}/train_history.pkl", mode="wb") as f:
        pickle.dump(train_losses, f)
    with open(f"./{model_type}/val_history.pkl", mode="wb") as f:
        pickle.dump(val_losses, f)
    print("Finished training.")
