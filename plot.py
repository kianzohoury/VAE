import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

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

        ax.set_xlabel("Epochs")
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
    with open(model_dir + "/class_results_val.pkl", mode="rb") as f:
        class_results = pickle.load(f)

    for class_idx in class_results:
        loss_term = "loss" if model_dir == "Autoencoder" else "recon_loss"
        ylabel = "MSE"
        fig, ax = plt.subplots(1, 1)
        latent_dims, results_arr = [], []
        for latent_num in sorted(class_results[class_idx][loss_term]):
            results_arr.append(class_results[class_idx][loss_term][latent_num])
            latent_dims.append(latent_num)

        ax.plot(latent_dims, results_arr, label=class_idx)
        ax.set_xlabel("Latent dimensions")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        # save figure
        fig.savefig(model_dir + f"/plots/class_results_MSE.jpg", dpi=300)
