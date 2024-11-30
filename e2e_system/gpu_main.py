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
from torch.utils.dlpack import to_dlpack
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

# GPU stuff.
from gputils.preproc import nvt_read_data, gpu_preproc, create_loader
from gputils.postproc import generate_large_data
from gputils.dataloaders import SequentialBatcher
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import cudf
import dask_cudf
import cupy as cp
import dask
import warnings
warnings.filterwarnings("ignore")


def train(config, X_train, num_continuous, num_categories, **kwargs):

    # Prepare data for interaction with torch VAE
    
    # Kwargs passed to data loader.
    data_loader = create_loader(X_train, batch_size=config.batch_size, **kwargs)

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


def main(config: Config):
    # No global client available, so have to init this ourselves.
    # Maybe something to do with VM environment?

    # TODO: Wrap this in a function maybe. Perhaps we have it throw errors
    # if it can't use GPUs.
    cluster = LocalCUDACluster(
        enable_cudf_spill=True,
        local_directory="/home/azureuser/datadrive/spill",
        threads_per_worker=1
    )
    client = Client(cluster)
    dask.config.set({"dataframe.backend": 'cudf'})

    print(f'Dashboard: {client.dashboard_link}')

    sys.stdout.flush()

    # if config.output_processed_data_path does not exist, create the directory
    if not os.path.exists(config.output_processed_data_save_dir):
        os.makedirs(config.output_processed_data_save_dir)
    if not os.path.exists(config.syn_data_save_dir):
        os.makedirs(config.syn_data_save_dir)

    # Read data. Calculate number of unique values -> determines categorical columns.
    # TODO: Maybe we add the option of specifying categoricals to save time.
    train_df, original_continuous_columns, original_categorical_columns = nvt_read_data(
        input_file_paths=config.input_data_path,
        categoricals=False,
        file_format='parquet'
    )
    # Save original columns (useful for data processing and formatting output).
    original_cols = list(train_df.columns)
    print("Continuous columns: ", original_continuous_columns)
    print("Categorical columns: ", original_categorical_columns)

    pre_proc_method = config.pre_proc_method
    print(f'Preprocessing Method: {pre_proc_method}')

    sys.stdout.flush()

    # TODO: Should this also return the names of the new columns?
    # Does the reverse_transformers() need to know the column names?

    # categories = cudf.DataFrame({'reliability': [250.0, 251.0, 252.0, 253.0, 254.0, 255.0]})

    (
        original_input_transformed,
        reordered_dataframe_columns,
        continuous_transformers,
        categorical_transformers,
        num_categories,
        num_continuous,
    ) = gpu_preproc(
        train_df,
        original_continuous_columns,
        original_categorical_columns,
        pre_proc_method=pre_proc_method,
        file_format='parquet',
        categories=None
    )

    # original_input_transformed.visualize(filename='task_graph.svg')
    
    X_train = original_input_transformed
    print(X_train.columns)
    print("Input data shape: ", X_train.shape[1])
    print(f'Partitions: {X_train.npartitions}')

    sys.stdout.flush()

    vae, encoder, decoder = train(
        config, 
        X_train, 
        num_continuous, 
        num_categories,
        threaded=True,
        keep_spill=False
    )

    load_vae = VAE(encoder, decoder)
    load_vae.load_state_dict(torch.load(config.filepath))

    # TODO: Figure out how to take generated data and
    # 1. Return it to original format.
    # 2. Write it to disk (idk if we want to do many files or one big file).
    generate_large_data(
        config, 
        load_vae, 
        X_train, 
        original_cols,
        reordered_dataframe_columns, 
        continuous_transformers, 
        categorical_transformers,
        original_continuous_columns,
        original_categorical_columns,
        num_continuous,
        batch_size=config.batch_size
    )
    print("Output written.")


if __name__ == '__main__':
    config = Config()

    # torch.autograd.set_detect_anomaly(True)
    # from torch.utils.viz._cycles import warn_tensor_cycles
    # warn_tensor_cycles()

    # TODO: Maybe make Config take command line args.
    config.pre_proc_method = 'standard'
    config.n_epochs = 1
    config.input_data_path = '/home/azureuser/datadrive/traffic/'
    # config.input_data_path = [
    #     # '/home/azureuser/datadrive/traffic/train_traffic',
    #     '/home/azureuser/datadrive/traffic/train_traffic-15000000.parquet',
    #     '/home/azureuser/datadrive/traffic/train_traffic-20000000.parquet',
    #     '/home/azureuser/datadrive/traffic/train_traffic-25000000.parquet',
    #     '/home/azureuser/datadrive/traffic/train_traffic-30000000.parquet',
    #     '/home/azureuser/datadrive/traffic/train_traffic-35000000.parquet',
    #     '/home/azureuser/datadrive/traffic/train_traffic-40000000.parquet',
    #     '/home/azureuser/datadrive/traffic/train_traffic-45000000.parquet',
    #     '/home/azureuser/datadrive/traffic/train_traffic-50000000.parquet',
    #     ]

    config.syn_data_save_dir = '/home/azureuser/datadrive/syn_traffic'
    config.filepath = 'shadowtraffic_vae_60g.pth'

    config.batch_size = 1024
    config.hidden_dim = 512
    config.latent_dim = 32

    config.patience = 3

    main(config)
