import os
import torch

class Config():
    """
    VAE model training configs
    """

    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.pre_proc_method = "GMM"
        # General training hyperparams
        self.batch_size = 1000
        self.latent_dim = 12
        self.hidden_dim = 512
        self.n_epochs = 2
        self.logging_freq = 1  # Number of epochs we should log the results to the user
        self.patience = 50  # How many epochs should we allow the model train to see if
        # improvement is made
        self.delta = 10  # The difference between elbo values that registers an improvement

        # Privacy params
        self.differential_privacy = False  # Do we want to implement differential privacy
        self.sample_rate = 0.1  # Sampling rate
        self.C = 1e16  # Clipping threshold any gradients above this are clipped
        self.noise_scale = 0.25  # Noise multiplier - influences how much noise to add
        self.target_eps = 2  # Target epsilon for privacy accountant
        self.target_delta = 1e-5  # Target delta for privacy accountant

        self.input_data_path = "data/all_data/cisco_data/network_traffic.csv"

        self.output_processed_data_save_dir = "data/all_data/vae_data_processed/"
        self.output_processed_data_path = os.path.join(self.output_processed_data_save_dir, 'network_traffic.csv')

        self.syn_data_save_dir = "syn_data/"
        self.syn_data_path = os.path.join(self.syn_data_save_dir, 'network_traffic.csv')

        self.model_save_dir = os.path.join(self.project_dir, 'saved_model/')    # Where to save the best model
        self.filepath = os.path.join(self.model_save_dir, 'cisco_network_traffic_vae.pth')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)


if __name__ == '__main__':
    config = Config()
    print(config.project_dir)
