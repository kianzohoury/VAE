
import pickle
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from . import utils

# set backend to SVG
matplotlib.use('svg')


def plot_epoch_train_validation(model_dir: str):
    """Plots epoch training and validation loss together."""
    with open(model_dir + "/train_history.pkl", mode="rb") as f:
        train_losses = pickle.load(f)
    with open(model_dir + "/val_history.pkl", mode="rb") as f:
        val_losses = pickle.load(f)

    for loss_term in val_losses:
        for latent_size in val_losses[loss_term]:
            fig, ax = plt.subplots(1, 1)
            ax.plot(val_losses[loss_term][latent_size], label="val")
            ax.plot(train_losses[loss_term][latent_size], label="train")

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
            fig.savefig(
                model_dir + f"/plots/train_val_latent_{latent_size}_{ylabel}.jpg",
                dpi=300
            )


def plot_epoch_history(model_dir: str, split: str = "val"):
    """Plots epoch training and validation losses."""
    model_type = Path(model_dir).stem.split("_")[0]
    with open(model_dir + f"/{split}_history.pkl", mode="rb") as f:
        losses = pickle.load(f)

    for loss_term in losses:
        fig, ax = plt.subplots(1, 1)
        for latent_size in losses[loss_term]:
            ax.plot(losses[loss_term][latent_size], label=latent_size)

        ax.set_xlabel("Epoch")
        if loss_term == "recon_loss":
            ylabel = "MSE"
        elif loss_term == "kl_loss":
            ylabel = "KL Divergence"
        else:
            ylabel = "ELBO" if "VAE" in model_dir else "MSE"
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        ax.set_title(
            f"Epoch {split[:1].upper() + split[1:]} History for {model_type}"
        )
        Path(model_dir + "/plots").mkdir(parents=True, exist_ok=True)
        # save figure
        fig.savefig(model_dir + f"/plots/{split}_{ylabel}.jpg", dpi=300)


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

        ax.scatter(latent_dims, results_arr, label=class_idx)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(ylabel)
    ax.legend(
        loc="upper center",
        ncol=len(class_results) // 2,
    )
    ax.set_title(
        f"Reconstruction Error for {model_type} Across Latent Dimensions"
    )
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

    img, label = next(iter(test_loader))
    for j in range(num_samples):
        ax[j][0].imshow(img[j].squeeze(0), cmap="gray")
        ax[j][0].axis("off")
    ax[0][0].set_title("Original")

    checkpoints = sorted(
        checkpoints, key=lambda file: int(file.stem.split("_")[-1])
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, checkpoint in enumerate(checkpoints):
        state_dict = torch.load(checkpoint, map_location=device)
        model_type = state_dict["config"].pop("model_type")

        # initialize model
        model = utils.init_model(model_type, **state_dict["config"]).to(device)
        num_latent = state_dict["config"]["num_latent"]
        num_classes = state_dict["config"]["num_classes"]
        model.load_state_dict(state_dict["model"])
        model.eval()

        if model_type == "ConditionalVAE":
            y = nn.functional.one_hot(label, num_classes)
            gen_img = model(img.to(device), y.to(device))
        else:
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


def plot_comparison(
    checkpoint: str,
    save_path: str = "./comparison.jpg",
    dataset: str = "mnist",
):
    """Plots generated images against their original images in a grid."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = utils.load_dataset(
        name=dataset,
        splits=("test"),
        val_size=0
    )

    # load model
    state_dict = torch.load(checkpoint, map_location=device)
    model_type = state_dict["config"].pop("model_type")
    model = utils.init_model(model_type, **state_dict["config"]).to(device)
    model.load_state_dict(state_dict["model"])
    model.eval()

    fig, ax = plt.subplots(
        nrows=2,
        ncols=10,
        constrained_layout=True,
        figsize=(10, 2)
    )
    ax[0][0].set_ylabel("Original")
    ax[1][0].set_ylabel("Generated")

    for class_idx in range(10):
        indices = []
        for idx, sample in enumerate(datasets["test"]):
            if sample[-1] == class_idx:
                indices.append(idx)

        # create dataloader
        test_loader = DataLoader(
            dataset=Subset(datasets["test"], indices),
            batch_size=1,
            shuffle=True,
            pin_memory=True
        )

        img, label = next(iter(test_loader))
        if dataset == "mnist":
            ax[0][class_idx].imshow(
                img.moveaxis(0, 3)[0], cmap="gray"
            )
        else:
            ax[0][class_idx].imshow(img.moveaxis(0, 3)[0], cmap=None)

        print(img.moveaxis(0, 3).shape)

        ax[0][class_idx].set_xticks([])
        ax[0][class_idx].set_yticks([])

        if model_type == "ConditionalVAE":
            y = nn.functional.one_hot(label, 10)
            gen_img = model(img.to(device), y.to(device))
        else:
            gen_img = model(img.to(device))
        if isinstance(gen_img, tuple):
            gen_img = gen_img[0]

        gen_img = gen_img.detach().cpu()
        gen_img = gen_img.moveaxis(0, 3)[0]

        if dataset == "mnist":
            ax[1][class_idx].imshow(gen_img, cmap="gray")
        else:
            ax[1][class_idx].imshow(gen_img, cmap=None)
        ax[1][class_idx].set_xticks([])
        ax[1][class_idx].set_yticks([])

    fig.suptitle(f"{'Digit' if dataset == 'mnist' else 'Class'}")
    fig.savefig(save_path, dpi=300)


def plot_new_samples(
    checkpoint: str,
    img_dim: Tuple[int, ...] = (28, 28, 1),
    classes: Optional[Tuple[int]] = None,
    num_samples: int = 1,
    num_cols: int = 10,
    title: str = "Generated Images",
    save_path: str = "./decodings.jpg"
):
    """Plots a grid of generated samples."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    state_dict = torch.load(checkpoint, map_location=device)
    model_type = state_dict["config"].pop("model_type")

    if "VAE" not in model_type:
        raise ValueError("Model must be a VAE.")
    elif model_type == "VAE":
        classes = None
    elif classes is None:
        classes = list(range(10))

    model = utils.init_model(model_type, **state_dict["config"]).to(device)
    model.load_state_dict(state_dict["model"])
    model.eval()

    # number of samples to generate
    num_samples = len(classes) * num_samples
    num_cols = min(num_cols, len(classes))

    fig, ax = plt.subplots(
        nrows=num_samples,
        ncols=num_cols,
        constrained_layout=True,
        figsize=(num_samples, num_cols)
    )

    # conditional VAE generation
    if classes is not None:
        for class_idx in classes:
            # sample z ~ N(0, 1)
            z = torch.randn(
                (num_samples, state_dict["config"]["num_latent"])
            ).to(device)

            y = nn.functional.one_hot(
                torch.Tensor([class_idx] * num_samples).long(),
                num_classes=10
            ).to(device)
            gen_img = model.decode(z, y).view(
                num_samples, *img_dim
            ).detach().cpu()

            for j in range(num_samples):
                ax[j][class_idx].imshow(gen_img[j], cmap="gray")
                ax[j][class_idx].axis("off")
                ax[j][class_idx].set_xticks([])
                ax[j][class_idx].set_yticks([])

            ax[0][class_idx].set_title(class_idx)
    else:
        # sample z ~ N(0, 1)
        z = torch.randn(
            (num_samples, state_dict["config"]["num_latent"])
        ).to(device)

        gen_img = model.decode(z).view(
            num_samples, *img_dim
        ).detach().cpu()

        num_rows = num_samples // num_cols
        for i in range(num_rows):
            for j in range(num_cols):
                ax[i][j].imshow(gen_img[num_cols * i + j], cmap="gray")
                ax[i][j].axis("off")

    fig.suptitle(title)
    fig.savefig(save_path, dpi=300)


def plot_tsne(
    checkpoint: str,
    save_path: str = "./tsne.jpg",
    dataset: str = "mnist",
    batch_size: int = 1024,
    num_workers: int = 4
):
    """Plots t-SNE."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = utils.load_dataset(
        name=dataset,
        splits=("test"),
        val_size=0
    )
    # create dataloader
    test_loader = DataLoader(
        dataset=datasets["test"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    # load model
    state_dict = torch.load(checkpoint, map_location=device)
    model_type = state_dict["config"].pop("model_type")
    model = utils.init_model(model_type, **state_dict["config"]).to(device)
    model.load_state_dict(state_dict["model"])
    model.eval()

    pca = PCA(n_components=50)
    tsne = TSNE(n_components=2)

    X_hat, Y = [], []
    model.eval()
    for idx, (img, label) in enumerate(test_loader, 0):
        bsize = img.shape[0]

        if model_type == "ConditionalVAE":
            y = nn.functional.one_hot(label, 10)
            gen_img = model(img.to(device), y.to(device))
        else:
            gen_img = model(img.to(device))
        if isinstance(gen_img, tuple):
            gen_img = gen_img[0]

        gen_img = gen_img.view(bsize, -1).detach().cpu()
        X_hat.append(gen_img)
        Y.append(label)

    X_hat = np.concatenate(X_hat, 0)
    Y = np.concatenate(Y, 0)

    # reduce dimensionality with PCA
    features = pca.fit_transform(X_hat)

    # reduce dimensionality again with t-SNE
    features = tsne.fit_transform(features)
    labeled_features = {idx: [] for idx in range(10)}
    for x_hat, y in zip(features, Y):
        labeled_features[y].append(x_hat)

    # plot embeddings
    fig, ax = plt.subplots(1, 1)
    for y, x in labeled_features.items():
        x_stack = np.stack(x, 0)
        ax.scatter(x_stack[:, 0], x_stack[:, 1], label=y)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.legend(loc="upper right")
    plt.title(f"t-SNE for {model_type}")
    fig.savefig(save_path, dpi=300)
