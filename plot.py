import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from . import utils

# set backend to SVG
matplotlib.use('svg')


def plot_epoch_validation(model_dir: str):
    """Plots epoch validation loss."""
    with open(model_dir + "/val_history.pkl", mode="rb") as f:
        val_losses = pickle.load(f)

    for loss_term in val_losses:
        fig, ax = plt.subplots(1, 1)
        for latent_size in val_losses[loss_term]:
            ax.plot(val_losses[loss_term][latent_size], label=latent_size)

        ax.set_xlabel("Epoch")
        if loss_term == "recon_loss":
            ylabel = "MSE"
        elif loss_term == "kl_loss":
            ylabel = "KL Divergence"
        else:
            ylabel = "ELBO" if "VAE" in model_dir else "MSE"
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        Path(model_dir + "/plots").mkdir(parents=True, exist_ok=True)
        # save figure
        fig.savefig(model_dir + f"/plots/validation_{ylabel}.jpg", dpi=300)


def plot_class_performance(model_dir: str):
    model_type = Path(model_dir).stem.split("_")[0]
    with open(model_dir + "/class_results_val.pkl", mode="rb") as f:
        class_results = pickle.load(f)

    fig, ax = plt.subplots(1, 1)
    loss_term = "loss" if model_type == "Autoencoder" else "recon_loss"
    ylabel = "MSE"
    for class_idx in class_results:
        latent_dims, results_arr = [], []
        for latent_num in sorted(class_results[class_idx][loss_term]):
            results_arr.append(
                class_results[class_idx][loss_term][latent_num][0]
            )
            latent_dims.append(latent_num)

        ax.plot(latent_dims, results_arr, label=class_idx)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    # save figure
    fig.savefig(model_dir + f"/plots/class_results_MSE.jpg", dpi=300)


def plot_reconstruction_grid(
    model_dir: str,
    dataset: str = "mnist",
    num_samples: int = 7
):
    """Plots a grid comparing images with generated reconstructions."""
    datasets = utils.load_dataset(
        name=dataset,
        splits=("test"),
        val_size=0
    )
    # create dataloader
    test_loader = DataLoader(
        dataset=datasets["test"],
        batch_size=num_samples,
        shuffle=True,
        pin_memory=True
    )
    checkpoints = list(Path(model_dir).rglob("*.pth"))
    fig, ax = plt.subplots(
        nrows=num_samples,
        ncols=len(checkpoints) + 1,
        constrained_layout=True,
        figsize=(10, 10)
    )

    img, _ = next(iter(test_loader))
    for j in range(num_samples):
        ax[j][0].imshow(img[j].squeeze(0), cmap="gray")
        ax[j][0].axis("off")
    ax[0][0].set_title("Original")

    checkpoints = sorted(
        checkpoints, key=lambda file: file.stem.split("_")[-1]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, checkpoint in enumerate(checkpoints):
        state_dict = torch.load(checkpoint, map_location=device)
        model_type = state_dict["config"].pop("model_type")

        # initialize model
        model = utils.init_model(model_type, **state_dict["config"]).to(device)
        num_latent = state_dict["config"]["num_latent"]
        model.load_state_dict(state_dict["model"])
        model.eval()
        gen_img = model(img.to(device))
        if isinstance(gen_img, tuple):
            gen_img = gen_img[0]
        gen_img = gen_img.detach().cpu()
        for j in range(num_samples):
            ax[j][i + 1].imshow(gen_img[j].squeeze(0), cmap="gray")
            ax[j][i + 1].axis("off")
        ax[0][i + 1].set_title(num_latent)

    fig.suptitle("Latent dimension")
    fig.savefig(f"{model_dir}/plots/reconstruction_grid.jpg")
