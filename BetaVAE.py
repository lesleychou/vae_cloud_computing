import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import typing as T


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class BetaVAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int = 20, beta: float = 1.0, lr=1e-4):
        """
        Beta-VAE implementation compatible with the VAE-torch structure.

        :param encoder: Encoder network.
        :param decoder: Decoder network.
        :param latent_dim: Dimensionality of the latent space.
        :param beta: Beta coefficient for the KL-divergence term.
        """
        super(BetaVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.num_categories = self.decoder.num_categories
        self.num_continuous = self.decoder.num_continuous
        self.beta = beta
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = encoder.device


    def encode(self, x: torch.Tensor):
        """
        Encodes the input into the latent space.

        :param x: Input tensor of shape (batch_size, ...).
        :returns: z_mean, z_log_var of shape (batch_size, latent_dim).
        """
        z_mean, z_log_var = self.encoder(x)
        return z_mean, z_log_var

    def reparameterize(self, z_mean: torch.Tensor, z_log_var: torch.Tensor):
        """
        Reparameterizes z using the reparameterization trick.

        :param z_mean: Mean of the latent distribution.
        :param z_log_var: Log variance of the latent distribution.
        :returns: Latent vector z.
        """
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z: torch.Tensor):
        """
        Decodes the latent vector back to the data space.

        :param z: Latent vector of shape (batch_size, latent_dim).
        :returns: Reconstructed data of shape (batch_size, ...).
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Full forward pass through the BetaVAE.

        :param x: Input tensor of shape (batch_size, ...).
        :returns: Reconstructed data, z_mean, z_log_var, z.
        """
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decode(z)
        return x_recon, z_mean, z_log_var, z

    def loss_function(self, x_recon: torch.Tensor, x: torch.Tensor, z_mean: torch.Tensor, z_log_var: torch.Tensor):
        """
        Computes the Beta-VAE loss.

        :param x_recon: Reconstructed data.
        :param x: Original data.
        :param z_mean: Mean of the latent distribution.
        :param z_log_var: Log variance of the latent distribution.
        :returns: Total loss, reconstruction loss, KL divergence loss.
        """
        # Reconstruction loss (MSE or BCE depending on the data type)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        # Total loss
        total_loss = recon_loss
        return total_loss, recon_loss, kl_loss

    # def generate(self, z: torch.Tensor):
    #     """
    #     Generates new data from latent vectors.

    #     :param z: Latent vectors of shape (batch_size, latent_dim).
    #     :returns: Generated data of shape (batch_size, ...).
    #     """
    #     return self.decode(z)
    
    def generate(self, N):
        z_samples = torch.randn_like(
            torch.ones((N, self.encoder.total_latent_dim)), device=self.device
        )
        x_gen = self.decoder(z_samples)
        x_gen_ = torch.ones_like(x_gen, device=self.device)
        i = 0

        for v in range(len(self.num_categories)):
            x_gen_[
            :, i: (i + self.num_categories[v])
            ] = torch.distributions.one_hot_categorical.OneHotCategorical(
                logits=x_gen[:, i: (i + self.num_categories[v])]
            ).sample()
            i = i + self.num_categories[v]

        x_gen_[:, -self.num_continuous:] = x_gen[
                                           :, -self.num_continuous:
                                           ]
        return x_gen_

    def sample(self, num_samples: int):
        """
        Samples new data points from the prior distribution.

        :param num_samples: Number of samples to generate.
        :param device: Device to generate the samples on.
        :returns: Generated data of shape (num_samples, ...).
        """
        z = torch.randn(num_samples, self.latent_dim).to( self.device)
        return self.generate(z)

    def reconstruct(self, x: torch.Tensor):
        """
        Reconstructs the input data.

        :param x: Input tensor of shape (batch_size, ...).
        :returns: Reconstructed data of shape (batch_size, ...).
        """
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decode(z)

    def train(self, dataloader, num_epochs=10, log_interval=100):
        """
        Trains the BetaVAE model.

        :param model: BetaVAE instance.
        :param dataloader: Dataloader providing the training data.
        :param optimizer: Optimizer for model parameters.
        :param device: Device to perform training on (e.g., 'cpu' or 'cuda').
        :param num_epochs: Number of training epochs.
        :param log_interval: Interval for logging the training progress.
        """

        for epoch in range(num_epochs):
            total_loss, total_recon_loss, total_kl_loss = 0, 0, 0
            for batch_idx, (data,) in enumerate(dataloader):
                data = data.to(self.device)

                # Forward pass
                x_recon, z_mean, z_log_var, _ = self(data)

                # Compute loss
                loss, recon_loss, kl_loss = self.loss_function(x_recon, data, z_mean, z_log_var)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate losses
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

                if batch_idx % log_interval == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}]: "
                        f"Loss = {loss.item():.4f}, Recon Loss = {recon_loss.item():.4f}, KL Loss = {kl_loss.item():.4f}"
                    )

            # Epoch summary
            avg_loss = total_loss / len(dataloader.dataset)
            avg_recon_loss = total_recon_loss / len(dataloader.dataset)
            avg_kl_loss = total_kl_loss / len(dataloader.dataset)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] Summary: "
                f"Avg Loss = {avg_loss:.4f}, Avg Recon Loss = {avg_recon_loss:.4f}, Avg KL Loss = {avg_kl_loss:.4f}"
            )


class Encoder(nn.Module):
    """Encoder, takes in x and outputs mu_z, sigma_z (diagonal Gaussian variational posterior assumed)"""

    def __init__(self, input_dim, latent_dim_hierarchy, hidden_dim=32, activation=nn.Tanh, device="gpu"):
        super().__init__()
        self.latent_dim_hierarchy = latent_dim_hierarchy
        self.total_latent_dim = sum(latent_dim_hierarchy)
        print("self.total_latent_dim: ", self.total_latent_dim)

        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Encoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Encoder: {device} specified, {self.device} used")
        output_dim = 2 * self.total_latent_dim
        print("Encoder output_dim: ", output_dim)
        # self.lstm = nn.LSTM(input_dim, hidden_dim)
        # self.linear = nn.Linear(hidden_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )


    def forward(self, x):
        outs = self.net(x)
        mu_z = outs[:, : self.total_latent_dim]
        logsigma_z = outs[:, self.total_latent_dim :]
        return mu_z, logsigma_z

class Decoder(nn.Module):
    """Decoder, takes in hierarchical latent dimensions and outputs reconstruction"""

    def __init__(
        self,
        latent_dim_hierarchy,
        num_continuous,
        num_categories=[0],
        hidden_dim=32,
        activation=nn.Tanh,
        device="gpu",
    ):
        super().__init__()

        self.latent_dim_hierarchy = latent_dim_hierarchy
        self.total_latent_dim = sum(latent_dim_hierarchy)
        output_dim = num_continuous + sum(num_categories)
        self.num_continuous = num_continuous
        self.num_categories = num_categories
        print("Decoder output_dim: ", output_dim)

        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Decoder: {device} specified, {self.device} used")

        self.net = nn.Sequential(
            nn.Linear(self.total_latent_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )
        # self.lstm = nn.LSTM(self.total_latent_dim, hidden_dim)
        # self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        return self.net(z)