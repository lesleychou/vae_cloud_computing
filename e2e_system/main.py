# Standard imports
import numpy as np
import pandas as pd
import torch
import math
import seaborn as sns  
import matplotlib.pyplot as plt
from config import Config
import os

# VAE is in other folder as well as opacus adapted library
import sys
sys.path.append("../")

# Opacus support for differential privacy
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE functions
from VAE import Decoder, Encoder, VAE

# Utility file contains all functions required to run notebook
from utils import (
    set_seed,
    mimic_pre_proc,
    plot_elbo,
    plot_likelihood_breakdown,
    plot_variable_distributions,
    reverse_transformers,
)
import warnings
warnings.filterwarnings("ignore")

def read_data(input_data_filepath, output_data_path):    
    # load data to pandas dataframe
    input_df = pd.read_csv(input_data_filepath)

    # remove the column "time" if it exists
    if 'time' in input_df.columns:
        input_df = input_df.drop(columns=['time'])

    # Find all column/features with categorical value
    original_categorical_columns = []
    categorical_len_count = 0
    for col in input_df:
        # Do not process the value
        if len(input_df[col].unique()) <= 10:
            original_categorical_columns.append(col)
            categorical_len_count += len(input_df[col].unique())

    original_continuous_columns = list(set(input_df.columns.values.tolist()) - set(original_categorical_columns))
    # # scale all continuous value with int log
    # for column in original_continuous_columns:
    #     input_df[column] = np.log1p(input_df[column])
    # input_df = input_df.round(0)
    input_df.to_csv(output_data_path, index=False)

    return input_df, original_continuous_columns, original_categorical_columns


def train(config, X_train, num_continuous, num_categories):

    # Prepare data for interaction with torch VAE
    Y = torch.Tensor(X_train)
    dataset = TensorDataset(Y)

    generator = None
    sample_rate = config.batch_size / len(dataset)
    data_loader = DataLoader(
        dataset,
        batch_sampler=UniformWithReplacementSampler(num_samples=len(dataset),
                                                    sample_rate=sample_rate,
                                                    generator=generator),
        pin_memory=True,
        generator=generator,
    )

    # Create VAE
    encoder = Encoder(X_train.shape[1], config.latent_dim, hidden_dim=config.hidden_dim)
    decoder = Decoder(config.latent_dim, num_continuous, hidden_dim=config.hidden_dim, num_categories=num_categories)

    vae = VAE(encoder, decoder)

    print(vae)

    if config.differential_privacy == False:
        (
            training_epochs,
            log_elbo,
            log_reconstruction,
            log_divergence,
            log_categorical,
            log_numerical,
        ) = vae.train(
            data_loader,
            n_epochs=config.n_epochs,
            logging_freq=config.logging_freq,
            patience=config.patience,
            delta=config.delta,
            filepath=config.filepath
        )

    elif config.differential_privacy == True:
        (
            training_epochs,
            log_elbo,
            log_reconstruction,
            log_divergence,
            log_categorical,
            log_numerical,
        ) = vae.diff_priv_train(
            data_loader,
            n_epochs=config.n_epochs,
            logging_freq=config.logging_freq,
            patience=config.patience,
            delta=config.delta,
            C=config.C,
            target_eps=config.target_eps,
            target_delta=config.target_delta,
            sample_rate=config.sample_rate,
            noise_scale=config.noise_scale,
            filepath=config.filepath
        )
        print(f"(epsilon, delta): {vae.get_privacy_spent(config.target_delta)}")

    return vae, encoder, decoder

def generate_diff_size(config, vae_model, X_train, input_df,
                       reordered_dataframe_columns, continuous_transformers, categorical_transformers, size=None):
    if not size:
        size = X_train.shape[0]
    # Generate synthetic data with X_train
    train_synthetic_sample = vae_model.generate(size)

    if torch.cuda.is_available():
        train_synthetic_sample = pd.DataFrame(
            train_synthetic_sample.cpu().detach().numpy(),
            columns=reordered_dataframe_columns
        )
    else:
        train_synthetic_sample = pd.DataFrame(
            train_synthetic_sample.detach().numpy(),
            columns=reordered_dataframe_columns
        )
    # Reverse the transformations

    train_synthetic_supp = reverse_transformers(
        synthetic_set=train_synthetic_sample,
        data_supp_columns=input_df.columns,
        cont_transformers=continuous_transformers,
        cat_transformers=categorical_transformers,
        pre_proc_method=config.pre_proc_method,
    )

    # remove all the negative value from generation
    train_synthetic_supp = train_synthetic_supp.clip(lower=0)
    train_synthetic_supp = train_synthetic_supp.round(0)

    return train_synthetic_supp

def main(config):
    # if config.output_processed_data_path does not exist, create the directory
    if not os.path.exists(config.output_processed_data_save_dir):
        os.makedirs(config.output_processed_data_save_dir)
    if not os.path.exists(config.syn_data_save_dir):
        os.makedirs(config.syn_data_save_dir)

    train_df, original_continuous_columns, original_categorical_columns = read_data(input_data_filepath=config.input_data_path,
                                                                                    output_data_path=config.output_processed_data_path)
    print("Continuous columns: ", original_continuous_columns)
    print("Categorical columns: ", original_categorical_columns)

    pre_proc_method = "standard"

    (
        original_input_transformed,
        original_input_original,
        reordered_dataframe_columns,
        continuous_transformers,
        categorical_transformers,
        num_categories,
        num_continuous,
    ) = mimic_pre_proc(train_df,
                       original_continuous_columns,
                       original_categorical_columns,
                       pre_proc_method=pre_proc_method)

    X_train = original_input_transformed
    print("Input data shape: ", X_train.shape)

    vae, encoder, decoder = train(config, X_train, num_continuous, num_categories)

    load_vae = VAE(encoder, decoder)
    load_vae.load_state_dict(torch.load(config.filepath))

    syn_generated_data = generate_diff_size(config, load_vae, X_train, train_df,
                                            reordered_dataframe_columns, continuous_transformers, categorical_transformers, size=None)
    syn_generated_data.to_csv(config.syn_data_path, index=False)

    # all_sizes = [10, 100, 1000, 5000, 10000, X_train.shape[0]]
    #
    # for size_i in all_sizes:
    #     syn_generated_data = generate_diff_size(config, load_vae, X_train, train_df,
    #                                             reordered_dataframe_columns, continuous_transformers,
    #                                             categorical_transformers, size=size_i)
    #     syn_generated_data.to_csv('syn_data/cisco_port_flap_dp/cisco_port_flap_{}.csv'.format(size_i), index=False)



if __name__ == '__main__':
    config = Config()
    main(config)
