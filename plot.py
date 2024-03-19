
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from . import utils

# set backend to SVG
matplotlib.use('svg')


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

        # set title
        ax.set_title(
            f"Epoch {split[:1].upper() + split[1:]} History for {model_type}"
        )
        Path(model_dir + "/plots").mkdir(parents=True, exist_ok=True)
        # save figure
        fig.savefig(model_dir + f"/plots/{split}_{ylabel}.jpg", dpi=300)


def plot_mse_by_class(model_dir: str) -> None:
    """Plots MSE values for each digit and latent size as a scatter plot."""
    model_type = Path(model_dir).stem.split("_")[0]

    # loads test results for each digit class
    with open(model_dir + "/class_results_val.pkl", mode="rb") as f:
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
    ax.legend(loc="upper center", ncol=2)
    ax.set_title(
        f"MSE for each Digit and Latent Size for {model_type}"
    )
    # save figure
    fig.savefig(model_dir + f"/plots/class_results_MSE.jpg", dpi=300)


def plot_reconstructed_digits(
    checkpoint: str,
    mnist_root: str = "./mnist",
    save_path: str = "./reconstructed_digits.jpg"
) -> None:
    """Plots generated images against their original images in a grid."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load dataset
    dataset = utils.load_dataset_splits(root=mnist_root, splits=["test"])



    # load model
    model = utils.load_from_checkpoint(checkpoint, device=device)
    model.eval()

    fig, ax = plt.subplots(
        nrows=2,
        ncols=10,
        constrained_layout=True,
        figsize=(10, 2)
    )
    ax[0][0].set_ylabel("Original")
    ax[1][0].set_ylabel("Recon")

    for digit in range(10):
        digit_subset = utils.filter_by_digit(dataset["test"], digit=digit)
        test_loader = utils.create_dataloaders(
            dataset_splits={"test": digit_subset},
            batch_size=1,
            num_workers=0
        )

        img, label = next(iter(test_loader["test"]))

        # plot original image
        ax[0][digit].imshow(img[0], cmap="gray")
        ax[0][digit].set_xticks([])
        ax[0][digit].set_yticks([])
        ax[0][digit].set_title(digit)

        if model.__name__ == "ConditionalVAE":
            y = nn.functional.one_hot(label, 10)
            gen_img = model(img.to(device), y.to(device))[0]
        elif model.__name__ == "VAE":
            gen_img = model(img.to(device))[0]
        else:
            gen_img = model(img.to(device))

        # plot reconstructed image
        gen_img = gen_img.detach().cpu()[0]
        ax[1][digit].imshow(gen_img, cmap="gray")
        ax[1][digit].set_xticks([])
        ax[1][digit].set_yticks([])

    # save figure
    fig.suptitle("Digit")
    fig.savefig(save_path, dpi=300)


def plot_generated_digits(
    checkpoint: str,
    samples_per_digit: 4,
    save_path: str = "./generated_digits.jpg"
):
    """Plots a sample_per_digit x 10 grid of generated samples."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = utils.load_from_checkpoint(checkpoint, device=device)
    model.eval()

    # number of samples to generate
    fig, ax = plt.subplots(
        nrows=samples_per_digit,
        ncols=10,
        constrained_layout=True,
        figsize=(10, samples_per_digit)
    )

    for digit in range(10):

        # conditional VAE generation
        if model.__name__ == "VAE":
            # sample z ~ N(0, 1)
            z = torch.randn((samples_per_digit, model.num_latent)).to(device)
            # create label vector
            y = nn.functional.one_hot(
                torch.Tensor([digit] * samples_per_digit).long(),
                num_classes=10
            ).to(device)

            # generate batch of new images
            gen_img = model.decode(z, y).view(
                samples_per_digit, 28, 28
            ).detach().cpu()

            # plot each generate image
            for j in range(samples_per_digit):
                ax[j][digit].imshow(gen_img[j], cmap="gray")
                ax[j][digit].axis("off")
                ax[j][digit].set_xticks([])
                ax[j][digit].set_yticks([])
                ax[0][digit].set_title(digit)

        # unconditional generation (AE and VAE)
        else:
            # sample z ~ N(0, 1)
            z = torch.randn((samples_per_digit, model.num_latent)).to(device)

            # generate batch of new images
            gen_img = model.decode(z).view(
                samples_per_digit, 28, 28
            ).detach().cpu()

            # plot each generate image
            for j in range(samples_per_digit):
                ax[j][digit].imshow(gen_img[j], cmap="gray")
                ax[j][digit].axis("off")

    fig.suptitle("Generated Digits")
    fig.savefig(save_path, dpi=300)


def plot_tsne_embeddings(
    checkpoint: str,
    mnist_root: str = "mnist",
    save_path: str = "./tsne_latent_space.jpg",
    batch_size: int = 1024,
    num_workers: int = 4
):
    """Plots t-SNE embeddings in order to visualize the latent space in 2D."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load dataset
    dataset = utils.load_dataset_splits(root=mnist_root, splits=["test"])
    test_loader = utils.create_dataloaders(
        dataset["test"], batch_size=batch_size, num_workers=num_workers
    )

    # load model
    model = utils.load_from_checkpoint(checkpoint, device=device)
    model.eval()

    # initialize PCA and t-SNE
    pca = PCA(n_components=min(model.num_latent, 20))
    tsne = TSNE(n_components=2)  # allows us to visualize latent space in 2D

    Z, Y = [], []
    model.eval()
    for _, (img, label) in enumerate(test_loader, 0):
        img = img.view(batch_size, -1).to(device)

        # extract latent representation
        if model.__name__ == "ConditionalVAE":
            y = nn.functional.one_hot(label, 10).long().to(device)
            mu, log_var = model.encode(img, y)
            z = model.reparameterize(mu, log_var)
        elif model.__name__ == "VAE":
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

    # reduce dimensionality again with t-SNE
    tsne_embeddings = tsne.fit_transform(pca_features)

    # group by digit class
    grouped_embeddings = {digit: [] for digit in range(10)}
    for (embedding, y) in zip(tsne_embeddings, Y):
        grouped_embeddings[y].append(embedding)

    # plot embeddings
    fig, ax = plt.subplots(1, 1)
    for (y, embeddings) in grouped_embeddings.items():
        embedding_stack = np.stack(embeddings, 0)
        dim1, dim2 = embedding_stack[:, 0], embedding_stack[:, 1]
        ax.scatter(dim1, dim2, label=y)

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.legend(loc="upper right")

    # save figure
    plt.title(f"t-SNE Embeddings in 2D for {model.__name__}")
    fig.savefig(save_path, dpi=300)
