
import pickle
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import colors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity

from . import utils

# set backend to SVG
matplotlib.use('svg')


def plot_mnist_tsne(
    mnist_root: str = "./mnist",
    split: str = "train",
    save_path: str = "./mnist_tsne.jpg",
    title: str = "Visualizing MNIST in 2D with t-SNE Features"
) -> None:

    # load mnist
    dataset = utils.load_mnist(root=mnist_root, split=split)

    X, Y = [], []
    for x, y in dataset:
        X.append(x)
        Y.append(y)
    X = np.stack(X).reshape(-1, 28 * 28)
    Y = np.stack(Y).reshape(-1, 1)

    # initialize PCA and t-SNE
    pca = PCA(n_components=50)
    tsne = TSNE(n_components=2, perplexity=50)

    # reduce feature dimensionality with PCA
    pca_features = pca.fit_transform(X)
    # reduce features further and generate clusters
    tsne_features = tsne.fit_transform(pca_features)

    # plot 2D t-SNE features
    fig, ax = plt.subplots(1, 1)
    cmap = plt.cm.rainbow
    norm = colors.BoundaryNorm(np.arange(0, 11, 1), cmap.N)

    img = ax.scatter(
        tsne_features[:, 0],
        tsne_features[:, 1],
        c=Y,
        cmap=cmap,
        norm=norm,
        s=5,
        edgecolor="none"
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_ticks(np.arange(0, 10, 1) + 0.5)
    cbar.set_ticklabels(list(range(10)))

    if title:
        plt.title(title)
    # save figure
    fig.savefig(save_path, bbox_inches="tight", dpi=300)


def plot_mnist_digits(
    mnist_root: str = "./mnist",
    split: str = "train",
    samples_per_digit: int = 5,
    save_path: str = "./mnist_digits.jpg",
    cmap: str = "gray",
) -> None:
    """Plots original MNIST digits in a grid."""
    # load mnist
    dataset = utils.load_mnist(root=mnist_root, split=split)
    n = 28

    # initialize empty array
    img_grid = np.zeros((samples_per_digit * n, 10 * n))
    for digit in range(10):
        digit_subset = utils.filter_by_digit(dataset, digit)
        for i in range(samples_per_digit):
            x, _ = digit_subset[i]
            # fill array
            img_grid[n * i: n * (i + 1), n * digit: n * (digit + 1)] = x

    fig, ax = plt.subplots(1, 1)
    # plot image
    ax.imshow(img_grid, cmap=cmap)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    # save figure
    fig.savefig(save_path, bbox_inches="tight", dpi=300)


def plot_epoch_history(model_dir: str, split: str = "val") -> None:
    """Plots epoch training or validation losses for each latent size."""
    model_type = Path(model_dir).stem.split("_")[0]

    # load losses
    with open(model_dir + f"/{split}_history.pkl", mode="rb") as f:
        losses = pickle.load(f)

    # iterate over loss types (e.g. KL, MSE, ELBO)
    for loss_term in losses:
        fig, ax = plt.subplots(1, 1)

        # iterate over latent sizes and plot results
        for latent_size in losses[loss_term]:
            ax.plot(losses[loss_term][latent_size], label=latent_size)

        # rename to a cleaner label
        if loss_term == "recon_loss":
            ylabel = "MSE"
        elif loss_term == "kl_loss":
            ylabel = "KL Divergence"
        else:
            ylabel = "ELBO" if "VAE" in model_dir else "MSE"

        # set x and y axis labels
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")

        split_ = "Validation" if split == "val" else "Training"
        title = f"{split_} {ylabel} Loss across Latent Sizes for {model_type}"
        # set title
        ax.set_title(title)

        Path(model_dir + "/plots").mkdir(parents=True, exist_ok=True)
        # save figure
        fig.savefig(model_dir + f"/plots/{split}_{ylabel}.jpg", dpi=300)


def plot_mse_by_class(model_dir: str) -> None:
    """Plots MSE values for each digit and latent size as a scatter plot."""
    model_type = Path(model_dir).stem.split("_")[0]

    # loads test results for each digit class
    with open(model_dir + "/class_results_test.pkl", mode="rb") as f:
        class_results = pickle.load(f)

    loss_term = "loss" if model_type == "Autoencoder" else "recon_loss"
    fig, ax = plt.subplots(1, 1)

    # iterate over each digit
    for digit in range(10):
        latent_dims, results_arr = [], []
        for latent_num in sorted(class_results[digit][loss_term]):
            results_arr.append(
                class_results[digit][loss_term][latent_num][0]
            )
            latent_dims.append(latent_num)

        ax.scatter(latent_dims, results_arr, label=digit)

    # set x, y labels, legend and title
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("MSE")
    ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    ax.set_title(
        f"MSE across Digits and Latent Sizes for {model_type}"
    )
    # save figure
    fig.savefig(
        model_dir + f"/plots/MSE_by_class.jpg", bbox_inches="tight", dpi=300
    )


def plot_reconstructed_digits(
    model_dir: str,
    mnist_root: str = "./mnist",
    save_path: str = "./reconstructed_digits.jpg",
    title: str = "Reconstructed Digits",
    seed: int = 0,
    cmap: str = "gray"
) -> None:
    """Plots generated images against their original images in a grid."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load dataset
    dataset = utils.load_dataset_splits(root=mnist_root, splits=["test"])

    checkpoints = list(Path(model_dir).rglob("*.pth"))
    checkpoints = sorted(
        checkpoints, key=lambda f: int(str(f.stem).split("_")[-1])
    )
    latent_dims = set()

    # initialize image grid
    n = 28
    img_grid = np.zeros((n * (1 + len(checkpoints)), 10 * n))
    for digit in range(10):
        digit_subset = utils.filter_by_digit(dataset["test"], digit)
        indices = list(range(len(digit_subset)))
        np.random.seed(seed)
        np.random.shuffle(indices)
        img, label = digit_subset[0]

        # fill array with real image
        img_grid[0: n, digit * n: (digit + 1) * n] = img.numpy()

        for j, checkpoint in enumerate(checkpoints):
            # load model
            model = utils.load_from_checkpoint(checkpoint, device=device)
            model_type = model.__class__.__name__
            num_latent = model.num_latent
            latent_dims.add(num_latent)
            model.eval()

            if model.__class__.__name__ == "ConditionalVAE":
                y = nn.functional.one_hot(label, 10)
                gen_img = model(img.to(device), y.to(device))[0]
            elif model.__class__.__name__ == "VAE":
                gen_img = model(img.to(device))[0]
            else:
                gen_img = model(img.to(device))

            # fill array with reconstructed image
            gen_img = gen_img.view(28, 28).detach().cpu().numpy()
            img_grid[(j + 1) * n: (j + 2) * n, n * digit: n * (digit + 1)] = gen_img

    fig, ax = plt.subplots(1, 1)
    ax.imshow(img_grid, cmap=cmap)
    ax.set_xticks([])

    # set y ticks
    y_ticks = n * np.arange(0, (1 + len(checkpoints)), 1) + (n // 2)
    y_labels = ["Original"] + [f"d={d}" for d in sorted(latent_dims)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    if title:
        plt.title(title)

    # save figure
    fig.savefig(save_path, bbox_inches="tight", dpi=300)


def plot_generated_digits(
    checkpoint: str,
    samples_per_digit: int = 4,
    save_path: str = "./generated_digits.jpg",
    title: str = "Generated Digits",
    cmap: str = "gray"
):
    """Plots a sample_per_digit x 10 grid of generated samples."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = utils.load_from_checkpoint(checkpoint, device=device)
    model_type = model.__class__.__name__
    num_latent = model.num_latent
    model.eval()

    # number of samples to generate
    fig, ax = plt.subplots(
        nrows=1,
        ncols=10,
        gridspec_kw={'wspace': 0, 'hspace': 0}
    )

    for digit in range(10):

        # conditional VAE generation
        if model.__class__.__name__ == "ConditionalVAE":
            # sample z ~ N(0, 1)
            z = torch.randn((samples_per_digit, model.num_latent)).to(device)
            # create label vector
            y = nn.functional.one_hot(
                torch.Tensor([digit] * samples_per_digit).long(),
                num_classes=10
            ).to(device)

            # generate batch of new images
            gen_img = model.decode(z, y).view(
                28 * samples_per_digit, 28
            ).detach().cpu()

            # plot generated image
            ax[digit].imshow(gen_img, cmap=cmap)
            ax[digit].axis("off")
            ax[digit].set_xticks([])
            ax[digit].set_yticks([])
            # ax[j][digit].set_aspect("equal")
            ax[digit].set_title(digit)

        # unconditional generation (AE and VAE)
        else:
            # sample z ~ N(0, 1)
            z = torch.randn((samples_per_digit, model.num_latent)).to(device)

            # generate batch of new images
            gen_img = model.decode(z).view(
                28 * samples_per_digit, 28
            ).detach().cpu()

            # plot generated image
            ax[digit].imshow(gen_img, cmap=cmap)
            # ax[j][digit].set_aspect("equal")
            ax[digit].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0, top=0.5)
    plt.tight_layout()
    if title:
        fig.suptitle(title)
    # save figure
    fig.savefig(save_path, dpi=300)


def plot_generated_digits_grid_2d(
    checkpoint: str,
    extent: Tuple[float,...],
    digit: int = 7,
    grid_size: int = 10,
    save_path: str = "./generated_digits.jpg",
    cmap: str = "gray"
) -> None:
    """Plots generated digits using z vectors from a grid in 2D space."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = utils.load_from_checkpoint(checkpoint, device=device)
    model_type = model.__class__.__name__
    num_latent = model.num_latent
    if num_latent != 2:
        raise ValueError("Model must have latent dimension of 2.")
    model.eval()

    fig, ax = plt.subplots(1, 1)
    x_coords = np.linspace(extent[0], extent[1], grid_size)
    y_coords = np.linspace(extent[2], extent[3], grid_size)
    z0, z1 = np.meshgrid(x_coords, y_coords)

    # initialize image grid to fill in
    img_grid = np.zeros((28 * grid_size, 28 * grid_size))

    # iterate over all z vectors in the uniform 2D space
    for i in range(grid_size):
        for j in range(grid_size):
            z = torch.Tensor([z0[i][j], z1[i][j]]).unsqueeze(0).to(device)

            # only use specified digit if conditioning is possible
            if model_type == "ConditionalVAE":
                # create label vector
                y = nn.functional.one_hot(
                    torch.Tensor([digit] * 1).long(),
                    num_classes=10
                ).to(device)
                gen_img = model.decode(z, y)
            else:
                gen_img = model.decode(z)

            gen_img = gen_img.squeeze(0).detach().cpu().view(28, 28)
            # fill image
            img_grid[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = gen_img

    ax.imshow(img_grid, cmap=cmap)
    ax.set_xticklabels(np.round(x_coords, 2))
    ax.set_yticklabels(np.round(y_coords, 2))

    fig.savefig("a.jpg")
    # save figure
    plt.title(
        f"Generated Digits for {model_type} over 2-d Latent Grid",
        fontsize=10
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=300)


def _run_pca(
    checkpoint: str,
    mnist_root: str = "mnist",
    split: str = "test",
    batch_size: int = 1024,
    num_workers: int = 4
) -> Dict[str, np.ndarray]:
    """Reduces latent space representations to 2D via PCA."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load dataset
    dataset = utils.load_dataset_splits(root=mnist_root, splits=split)
    test_loader = utils.create_dataloaders(
        dataset, batch_size=batch_size, num_workers=num_workers
    )

    # load model
    model = utils.load_from_checkpoint(checkpoint, device=device)
    num_latent = model.num_latent
    model.eval()

    # initialize PCA
    pca = PCA(n_components=min(model.num_latent, 2))

    Z, Y = [], []
    model.eval()
    for _, (img, label) in enumerate(test_loader[split], 0):
        img = img.view(img.shape[0], -1).to(device)

        # extract latent representation
        if model.__class__.__name__ == "ConditionalVAE":
            y = nn.functional.one_hot(label, 10).long().to(device)
            mu, log_var = model.encode(img, y)
            z = model.reparameterize(mu, log_var)
        elif model.__class__.__name__ == "VAE":
            mu, log_var = model.encode(img)
            z = model.reparameterize(mu, log_var)
        else:
            z = model.encode(img)

        Z.append(z.detach().cpu().numpy())
        Y.append(label)

    Z = np.concatenate(Z, 0)
    Y = np.concatenate(Y, 0)

    # reduce dimensionality with PCA
    pca_features = pca.fit_transform(Z)

    # group by digit class
    grouped_features = {digit: [] for digit in range(10)}
    for (feature, y) in zip(pca_features, Y):
        grouped_features[y].append(feature)

    # stack features into single array
    for (y, feature) in grouped_features.items():
        feature_stack = np.stack(feature, 0)
        grouped_features[y] = feature_stack
    return grouped_features


def plot_latent_space_scatter_2d(
    checkpoint: str,
    mnist_root: str = "mnist",
    split: str = "test",
    save_path: str = "./latent_space_scatter_2d.jpg",
    title: str = "Visualizing Latent Space with PCA",
    batch_size: int = 1024,
    num_workers: int = 4
):
    """Plots latent space representations as a 2D scatter plot."""
    model = utils.load_from_checkpoint(checkpoint)
    num_latent = model.num_latent

    # get reduced features
    grouped_features = _run_pca(
        checkpoint=checkpoint,
        mnist_root=mnist_root,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # plot 2D features
    fig, ax = plt.subplots(1, 1)
    cmap = plt.cm.rainbow
    norm = colors.BoundaryNorm(np.arange(0, 11, 1), cmap.N)

    stacked_features = []
    stacked_labels = []
    for (y, features) in grouped_features.items():
        stacked_features.append(features)
        stacked_labels.append([y] * len(features))

    stacked_features = np.concatenate(stacked_features)
    stacked_labels = np.concatenate(stacked_labels)

    img = ax.scatter(
        stacked_features[:, 0],
        stacked_features[:, 1],
        c=stacked_labels,
        cmap=cmap,
        norm=norm,
        s=5,
        edgecolor="none"
    )

    ax.set_xlabel("Principal Component 1" if num_latent != 2 else "z1")
    ax.set_ylabel("Principal Component 2" if num_latent != 2 else "z2")
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_ticks(np.arange(0, 10, 1) + 0.5)
    cbar.set_ticklabels(list(range(10)))

    if title:
        plt.title(title)
    # save figure
    fig.savefig(save_path, bbox_inches="tight", dpi=300)


def plot_latent_space_kde_1d(
    checkpoint: str,
    mnist_root: str = "mnist",
    save_path: str = "./latent_space_kde_1d.jpg",
    batch_size: int = 1024,
    num_workers: int = 4,
    num_bins: int = 100,
    bandwidth: float = 0.25
):
    """Plots a 1D histogram of the latent space along with a KDE."""
    model = utils.load_from_checkpoint(checkpoint)
    num_latent = model.num_latent

    # get reduced features
    grouped_features = _run_pca(
        checkpoint=checkpoint,
        mnist_root=mnist_root,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # collapse grouped features into one array
    features = np.concatenate(
        list(grouped_features.values()), axis=0
    ).reshape(-1, 1)

    # plot histogram
    fig, ax = plt.subplots(1, 1)
    density, bins, _ = ax.hist(
        features,
        bins=num_bins,
        label="True Distribution",
        density=True,
        alpha=0.25
    )

    # estimate PDF with KDE with a Gaussian kernel
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(features)

    # plot KDE
    min_x, max_x = min(bins), max(bins)
    eval_points = np.linspace(min_x, max_x, num=1000).reshape(-1, 1)
    log_likelihood = kde.score_samples(eval_points)
    densities = np.exp(log_likelihood)
    ax.plot(eval_points, densities, label="KDE")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Density")
    plt.legend()

    # save figure
    plt.title(
        f"KDE of {num_latent}-d Latent Space reduced to "
        f"1-d with PCA for {model.__class__.__name__}"
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
