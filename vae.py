
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Vanilla autoencoder implemented with MLPs."""
    def __init__(
        self,
        num_features: int = 28 * 28,
        num_hidden: int = 28 * 28,
        num_latent: int = 100
    ):
        super(Autoencoder, self).__init__()
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_latent = num_latent

        self.encoder = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_latent),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_latent, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_features),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compresses x to latent representation, i.e. z = enc(x)."""
        z = self.encoder(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs x from latent representation. i.e. x_hat = dec(z)"""
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates x_hat from x, i.e. x_hat = dec(enc(x))."""
        z = self.encode(x.view(x.size()[0], -1))  # flatten image tensor
        x_hat = self.decode(z).view(x.size())  # reshape again as image
        return x_hat

    def training_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes the reconstruction loss."""
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        return {"loss": loss}


class VAE(nn.Module):
    """Simple variational autoencoder (VAE) implemented with MLPs."""
    def __init__(
        self,
        num_features: int = 28 * 28,
        num_hidden: int = 28 * 28,
        num_latent: int = 100,
        kl_weight: float = 1.0
    ):
        super(VAE, self).__init__()
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_latent = num_latent
        self.kl_weight = kl_weight

        self.encoder = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_latent, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_features),
            nn.Sigmoid()
        )

        self.mu = nn.Linear(num_hidden, num_latent)
        self.log_var = nn.Linear(num_hidden, num_latent)
        self.eps = torch.distributions.Normal(0, 1)

        # Speed up sampling by utilizing GPU.
        if torch.cuda.is_available():
            self.eps.loc = self.eps.loc.cuda()
            self.eps.scale = self.eps.scale.cuda()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps input x to latent distribution p_theta(z|x)."""
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Performs reparameterization trick: z = eps * std + mu."""
        std = torch.exp(0.5 * log_var)  # equivalent to sqrt(exp(log_var))
        eps = self.eps.sample(mu.shape)  # sample eps ~ N(0, 1)
        z = eps * std + mu
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Maps latent variable z to distribution q_phi(x|z)."""
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Generates x_hat from x, i.e. x_hat = dec(enc(x))."""
        mu, log_var = self.encode(x.view(x.size()[0], -1))  # flatten again
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z).view(x.size())  # reshape again as an image
        return x_hat, mu, log_var

    def training_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes the loss as KL + reconstruction loss."""
        x_hat, mu, log_var = self.forward(x)
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        kl_loss = kl_div_loss(mu, log_var)
        loss = recon_loss + self.kl_weight * kl_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }


class ConditionalVAE(nn.Module):
    """Simple variational autoencoder using MLPs."""
    def __init__(
        self,
        num_classes: int = 10,
        num_features: int = 28 * 28,
        num_hidden: int = 28 * 28,
        num_latent: int = 100,
        kl_weight: float = 1.0
    ):
        super(ConditionalVAE, self).__init__()
        self.num_classes = num_classes
        # increase feature and hidden dimensionality by size of
        # the label vector
        num_features += num_classes
        num_hidden += num_classes
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_latent = num_latent
        self.kl_weight = kl_weight

        self.encoder = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_latent + num_classes, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_features - num_classes),
            nn.Sigmoid()
        )

        self.mu = nn.Linear(num_hidden, num_latent)
        self.log_var = nn.Linear(num_hidden, num_latent)
        self.eps = torch.distributions.Normal(0, 1)

        # Speed up sampling by utilizing GPU.
        if torch.cuda.is_available():
            self.eps.loc = self.eps.loc.cuda()
            self.eps.scale = self.eps.scale.cuda()

    def encode(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps input x, y to latent distribution p_theta(z|x, y)."""
        x = torch.cat([x, y], dim=1)  # combine x and y
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Performs reparameterization trick: z = eps * std + mu."""
        std = torch.exp(0.5 * log_var)
        eps = self.eps.sample(mu.shape) # sample eps ~ N(0, 1)
        z = eps * std + mu
        return z

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Maps latent variable z to distribution q_phi(x|z, y)."""
        z = torch.cat([z, y], dim=1)  # combine z and y
        x_hat = self.decoder(z)
        return x_hat

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Generates x_hat from x and y, i.e. x_hat = dec(enc(x, y))."""
        mu, log_var = self.encode(x.view(x.size()[0], -1), y) # flatten again
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z, y).view(x.size()) # reshape again as an image
        return x_hat, mu, log_var

    def training_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Computes the loss as KL + reconstruction loss."""
        # vectorize label
        y = nn.functional.one_hot(y, self.num_classes).to(x.device)
        x_hat, mu, log_var = self.forward(x, y)
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        kl_loss = kl_div_loss(mu, log_var)
        loss = recon_loss + self.kl_weight * kl_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }


def kl_div_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Kullback-Leibler Divergence loss."""
    return -0.5 * torch.sum(1 + log_var - (mu ** 2) - log_var.exp())
