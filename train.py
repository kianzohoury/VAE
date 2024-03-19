import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from . import utils, vae

###############################################################################
#                                 Testing                                     #
###############################################################################


def test(model: nn.Module, test_loader: DataLoader) -> torch.Tensor:
    """Runs testing (or validation) and returns the loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_loss = defaultdict(float)
    model.eval()
    for idx, (img, label) in enumerate(test_loader, 1):
        img = img.to(device)

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
    model_dir: str,
    dataset: str = "mnist",
    is_test: bool = False,
    batch_size: int = 1024,
    num_workers: int = 4
) -> None:
    """Tests models separately for each class (e.g. digits in MNIST)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_test:
        datasets = utils.load_dataset(
            name=dataset,
            splits=("test"),
            val_size=0
        )
    else:
        # use validation set (necessary for model selection, and to keep
        # test set independent of this process)
        datasets = utils.load_dataset(
            name=dataset,
            splits=("train", "val"),
            val_size=0.1
        )
    test_dataset = datasets["test"] if is_test else datasets["val"]
    test_losses = defaultdict(partial(defaultdict, partial(defaultdict, list)))

    for class_idx in range(10):
        indices = []
        for idx, sample in enumerate(test_dataset):
            if sample[-1] == class_idx:
                indices.append(idx)

        # create dataloader
        test_loader = DataLoader(
            dataset=Subset(test_dataset, indices),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
        )

        print(f"Starting testing for class {class_idx}...")
        for checkpoint in list(Path(model_dir).rglob("*.pth")):
            state_dict = torch.load(checkpoint, map_location=device)
            model_type = state_dict["config"].pop("model_type")
            # initialize model
            model = utils.init_model(
                model_type, **state_dict["config"]
            ).to(device)
            num_latent = state_dict["config"]["num_latent"]
            model.load_state_dict(state_dict["model"])

            # test
            test_loss = test(model, test_loader)
            for loss_term, loss_val in test_loss.items():
                test_losses[class_idx][loss_term][num_latent].append(loss_val)
                print(
                    f"{model_type} (z-dim={num_latent}), "
                    f"{loss_term}: {round(loss_val, 3)}"
                )

    # save test results
    print("Saving test results...")
    pkl_filename = f"{model_dir}/class_results_test.pkl" if is_test else \
        f"{model_dir}/class_results_val.pkl"
    with open(pkl_filename, mode="wb") as f:
        pickle.dump(test_losses, f)
    print("Finished testing.")


###############################################################################
#                                 Training                                    #
###############################################################################

def train(
    model_type: str = "VAE",
    dataset: str = "mnist",
    validate: bool = True,
    latent_size: Union[int, Tuple] = 20,
    num_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 1024,
    num_workers: int = 4,
    output_dir: str = "./output"
) -> None:
    """Trains given model type for each latent representation size."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = utils.load_dataset(
        name=dataset,
        splits=("train", "val") if validate else ("train",),
        val_size=0 if not validate else 0.1)

    # create dataloaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    if validate:
        val_loader = DataLoader(
            datasets["val"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
    else:
        val_loader = None

    # store configuration args for easy model loading
    config = {
        "num_features": 28 * 28 if dataset == "mnist" else 32 * 32 * 3,
        "model_type": model_type
    }
    # number of classes is the same for MNIST and CIFAR10
    if model_type == "ConditionalVAE":
        config["num_classes"] = 10

    # keep track of metrics across all models
    train_losses = defaultdict(partial(defaultdict, list))
    val_losses = defaultdict(partial(defaultdict, list))

    # create output directory to store checkpoints and results
    output_path = Path(f"{output_dir}/{model_type}")
    output_path.mkdir(exist_ok=True, parents=True)

    # simple hyperparameter search over latent dimensions
    latent_search_space = [latent_size] if isinstance(latent_size, int) \
        else list(latent_size)
    for num_latent in latent_search_space:

        # initialize model and optimizer
        model_config = dict(config)
        model_config["num_latent"] = num_latent
        model = utils.init_model(**model_config).to(device)
        optim = AdamW(model.parameters(), lr)
        print(f"Starting training for z-dim={num_latent}.")

        model.train()
        for epoch in range(num_epochs):
            total_loss = defaultdict(float)
            with tqdm(train_loader) as tq:
                tq.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
                for idx, (img, label) in enumerate(tq, 1):
                    img = img.to(device)

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
                validation_loss = validate(model, val_loader)
                for loss_term, loss_val in validation_loss.items():
                    val_losses[loss_term][num_latent].append(loss_val)
                    print(f"Val: {loss_term}={round(loss_val, 3)}")

        print("Saving model...")
        torch.save(
            {"model": model.cpu().state_dict(), "config": model_config},
            f=f"{str(output_path)}_latent_{num_latent}.pth",
        )

    # save epoch history
    print("Saving loss history...")
    with open(f"{str(output_path)}/train_history.pkl", mode="wb") as f:
        pickle.dump(train_losses, f)
    with open(f"{str(output_path)}/val_history.pkl", mode="wb") as f:
        pickle.dump(val_losses, f)
    print("Finished training.")
