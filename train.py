
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from . import utils, vae

###############################################################################
#                                 Testing                                     #
###############################################################################


def test(model: nn.Module, test_loader: DataLoader) -> Dict[str, torch.Tensor]:
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


def test_by_class(
    model_dir: str,
    mnist_root: str = "./mnist",
    batch_size: int = 1024,
    num_workers: int = 4
) -> None:
    """Tests models separately for each class (i.e. digits in MNIST)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load dataset
    dataset = utils.load_dataset_splits(root=mnist_root, splits=["test"])
    test_losses = defaultdict(partial(defaultdict, partial(defaultdict, list)))

    for class_idx in range(10):
        digit_subset = utils.filter_by_digit(dataset["test"], digit=class_idx)
        test_loader = utils.create_dataloaders(
            dataset_splits={"test": digit_subset},
            batch_size=batch_size,
            num_workers=num_workers
        )

        print(f"Starting testing for class {class_idx}...")
        for checkpoint in list(Path(model_dir).rglob("*.pth")):
            model = utils.load_from_checkpoint(checkpoint, device=device)
            num_latent = model.num_latent

            # test
            test_loss = test(model, test_loader["test"])
            for loss_term, loss_val in test_loss.items():
                test_losses[class_idx][loss_term][num_latent].append(loss_val)
                print(
                    f"{model.__name__} (z-dim={num_latent}), "
                    f"{loss_term}: {round(loss_val, 3)}"
                )

    # save test results
    print("Saving test results...")
    with open(f"{model_dir}/class_results_test.pkl" , mode="wb") as f:
        pickle.dump(test_losses, f)
    print("Finished testing.")


###############################################################################
#                                 Training                                    #
###############################################################################

def train(
    mnist_root: str = "./mnist",
    model_type: str = "ConditionalVAE",
    validate: bool = True,
    latent_size: Union[int, Tuple] = 20,
    kl_weight: float = 1.0,
    num_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 1024,
    num_workers: int = 4,
    output_dir: str = "./output"
) -> None:
    """Trains given model type for each latent representation size."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load dataset
    dataset = utils.load_dataset_splits(
        root=mnist_root,
        splits=["train", "val"] if validate else ["train"],
        val_size=0 if not validate else 0.1
    )
    # get dataloaders
    dataloaders = utils.create_dataloaders(
        dataset_splits=dataset, batch_size=batch_size, num_workers=num_workers
    )

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

        config = {
            "model_type": model_type,
            "latent_size": latent_size,
            "kl_weight": kl_weight  # only applies for ConditionalVAE
        }

        # initialize model and optimizer
        model = utils.init_model(model_type=model_type, config=config)
        optim = AdamW(model.parameters(), lr)
        print(f"Starting training for z-dim={num_latent}.")

        model.train()
        for epoch in range(num_epochs):
            total_loss = defaultdict(float)
            with tqdm(dataloaders["train"]) as tq:
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
                validation_loss = test(model, dataloaders["val"])
                for loss_term, loss_val in validation_loss.items():
                    val_losses[loss_term][num_latent].append(loss_val)
                    print(f"Val: {loss_term}={round(loss_val, 3)}")

        print("Saving model...")
        torch.save(
            {"model": model.cpu().state_dict(), "config": config},
            f=f"{str(output_path)}/{model_type}_latent_{num_latent}.pth",
        )

    # save epoch history
    print("Saving loss history...")
    with open(f"{str(output_path)}/train_history.pkl", mode="wb") as f:
        pickle.dump(train_losses, f)
    with open(f"{str(output_path)}/val_history.pkl", mode="wb") as f:
        pickle.dump(val_losses, f)
    print("Finished training.")
