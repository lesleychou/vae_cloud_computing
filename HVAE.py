from random import gauss
from pandas import Categorical
import torch
import torch.nn as nn

from opacus import PrivacyEngine

# from torch.distributions.bernoulli import Bernoulli
from torch.distributions import StudentT, Normal

from tqdm import tqdm
import pandas as pd


class HierarchicalVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim_hierarchy, lr=1e-3):
        super().__init__()
        self.encoder = encoder.to(encoder.device)
        self.decoder = decoder.to(decoder.device)
        self.latent_dim_hierarchy = latent_dim_hierarchy
        self.device = encoder.device
        self.num_categories = self.decoder.num_categories
        self.num_continuous = self.decoder.num_continuous
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=0, lr=lr)
        self.lr = lr
        self.hierarchy_indices = self._create_hierarchy_indices()

    def _create_hierarchy_indices(self):
        # Create indices to split the latent space into hierarchical groups
        indices = []
        start = 0
        for dim in self.latent_dim_hierarchy:
            indices.append((start, start + dim))
            start += dim
        return indices

    def get_hierarchical_latent(self, mu_z):
        # Split the latent variables into hierarchical groups
        return [mu_z[:, start:end] for start, end in self.hierarchy_indices]

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
    
    def loss(self, X):
        mu_z, logsigma_z = self.encoder(X)

        # p = Normal(torch.zeros_like(mu_z), torch.ones_like(mu_z))
        # q = Normal(mu_z, torch.exp(logsigma_z))
        # ignore 
        divergence_loss = 0

        z_samples = mu_z

        x_recon = self.decoder(z_samples)

        categoric_loglik = 0
        if sum(self.num_categories) != 0:
            i = 0

            for v in range(len(self.num_categories)):
                # import pdb; pdb.set_trace()

                categoric_loglik += -torch.nn.functional.cross_entropy(
                    x_recon[:, i : (i + self.num_categories[v])],
                    torch.max(X[:, i : (i + self.num_categories[v])], 1)[1],
                ).sum()
                i = i + self.decoder.num_categories[v]

        gauss_loglik = (
            Normal(
                loc=x_recon[:, -self.num_continuous:],
                scale=torch.ones_like(x_recon[:, -self.num_continuous:]),
            )
            .log_prob(X[:, -self.num_continuous:])
            .sum()
        )

        # Use StudentT for heavy-tailed reconstruction
        # df = 3.0  # Degrees of freedom (lower = heavier tails, >2 ensures finite variance)
        # t_dist = StudentT(df, loc=x_recon[:, -self.num_continuous:], scale=torch.ones_like(x_recon[:, -self.num_continuous:]))
        # t_loglik = t_dist.log_prob(X[:, -self.num_continuous:]).sum()

        reconstruct_loss = -(categoric_loglik + gauss_loglik)

        elbo = reconstruct_loss

        return (elbo, reconstruct_loss, divergence_loss, categoric_loglik, gauss_loglik)


    def train(self, 
              x_dataloader, 
              n_epochs, 
              logging_freq=1, 
              filepath=None):
        log_elbo = []
        log_reconstruct = []
        log_divergence = []
        log_cat_loss = []
        log_num_loss = []

        for epoch in range(n_epochs):
            train_loss = 0.0
            reconstruction_epoch_loss = 0.0
            categorical_epoch_reconstruct = 0.0
            numerical_epoch_reconstruct = 0.0

            for batch_idx, (Y_subset,) in enumerate(tqdm(x_dataloader)):
                self.optimizer.zero_grad()
                elbo, reconstruct_loss, divergence_loss, categorical_reconstruct, numerical_reconstruct = self.loss(Y_subset.to(self.device))
                elbo.backward()
                self.optimizer.step()

                train_loss += elbo.item()
                reconstruction_epoch_loss += reconstruct_loss.item()
                categorical_epoch_reconstruct += categorical_reconstruct.item()
                numerical_epoch_reconstruct += numerical_reconstruct.item()

                # Extract hierarchical latent variables
                mu_z, _ = self.encoder(Y_subset.to(self.device))
                hierarchical_latent = self.get_hierarchical_latent(mu_z)

                # Optionally: Save hierarchical latent variables
                if filepath:
                    torch.save(hierarchical_latent, f"{filepath}_epoch_{epoch}.pt")

            log_elbo.append(train_loss)
            log_reconstruct.append(reconstruction_epoch_loss)
            log_cat_loss.append(categorical_epoch_reconstruct)
            log_num_loss.append(numerical_epoch_reconstruct)

            if epoch % logging_freq == 0:
                print(
                    f"\tEpoch: {epoch:2}. Elbo: {train_loss:11.2f}. Reconstruction Loss: {reconstruction_epoch_loss:11.2f}. Categorical Loss: {categorical_epoch_reconstruct:11.2f}. Numerical Loss: {numerical_epoch_reconstruct:11.2f}"
                )

            # visualize the reconstruction loss

            # # Optionally: Describe dataset with hierarchical latent variables
            # if filepath:
            #     latent_embeddings = torch.cat([self.vae.encoder(Y_subset.to(self.vae.device))[0] for (Y_subset,) in x_dataloader], dim=0)
            #     bayesian_networks = self.describe_dataset_with_hierarchical_latent(latent_embeddings, self.vae.hierarchy_indices, self.k, self.epsilon)
            #     torch.save(bayesian_networks, f"{filepath}_bayesian_networks_epoch_{epoch}.pt")

        return (
            n_epochs,
            log_elbo,
            log_reconstruct,
            log_divergence,
            log_cat_loss,
            log_num_loss,
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


# Example usage:
# model = HierarchicalVAE(encoder, decoder, latent_dim_hierarchy)
# model = load_model_state(model, 'path_to_checkpoint.pth')