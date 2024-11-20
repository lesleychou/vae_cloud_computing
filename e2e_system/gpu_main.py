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

# GPU stuff.
from gputils.preproc import nvt_read_data, gpu_preproc, create_loader
from gputils.postproc import gpu_reverse_transformers
from gputils.dataloaders import SequentialBatcher
from dask.distributed import Client
import cudf
import dask_cudf
import cupy as cp
import warnings
warnings.filterwarnings("ignore")


def train(config, X_train, num_continuous, num_categories):

    # Prepare data for interaction with torch VAE
    
    # TODO: Maybe I implement UniformWithReplacement sampling later? Idk 
    # how important that is.
    data_loader = create_loader(X_train, batch_size=config.batch_size)

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
        size = len(X_train)
    # Generate synthetic data with X_train
    train_synthetic_sample = vae_model.generate(size)
    

    if torch.cuda.is_available():
        # trying to 
        # synthetic_shape = train_synthetic_sample.shape
        # synthetic_dtype = train_synthetic_sample.dtype

        # dtype_map = {
        #     torch.float32: cp.float32,
        #     torch.float64: cp.float64,
        #     torch.int32: cp.int32,
        #     torch.int64: cp.int64,
        # }

        # synthetic_cupy_dtype = dtype_map[synthetic_dtype]

        # train_synthetic_sample = cp.ndarray(
        #     shape=synthetic_shape,                   
        #     dtype=synthetic_cupy_dtype,               
        #     memptr=cp.cuda.MemoryPointer(   
        #         cp.cuda.Memory(train_synthetic_sample.data_ptr()), 0
        #     )
        # )
        train_synthetic_sample = cp.asarray(train_synthetic_sample.cpu().detach().numpy())
        train_synthetic_sample = cudf.DataFrame(
            train_synthetic_sample,
            columns=reordered_dataframe_columns
        )
        train_synthetic_sample = dask_cudf.from_cudf(train_synthetic_sample, npartitions=2560)
    else:
        train_synthetic_sample = pd.DataFrame(
            train_synthetic_sample.detach().numpy(),
            columns=reordered_dataframe_columns
        )
    # Reverse the transformations

    train_synthetic_supp = gpu_reverse_transformers(
        synthetic_set=train_synthetic_sample,
        data_supp_columns=input_df.columns,
        cont_transformers=continuous_transformers,
        cat_transformers=categorical_transformers,
        pre_proc_method=config.pre_proc_method,
    )

    # remove all the negative value from generation
    # train_synthetic_supp = train_synthetic_supp.clip(lower=0)
    # train_synthetic_supp = train_synthetic_supp.round(0)
    train_synthetic_supp = train_synthetic_supp.map_partitions(
        lambda df: df.clip(lower=0).round(0), meta=train_synthetic_supp._meta
    )

    return train_synthetic_supp

def main(config):
    # No global client available, so have to init this ourselves.
    # Maybe something to do with VM environment?
    client = Client()
    # if config.output_processed_data_path does not exist, create the directory
    if not os.path.exists(config.output_processed_data_save_dir):
        os.makedirs(config.output_processed_data_save_dir)
    if not os.path.exists(config.syn_data_save_dir):
        os.makedirs(config.syn_data_save_dir)

    # TODO: Make this function accept glob of file paths.
    train_df, original_continuous_columns, original_categorical_columns = nvt_read_data(input_file_paths=[config.input_data_path for _ in range(2)],
                                                                                    output_file_path=config.output_processed_data_path)
    print("Continuous columns: ", original_continuous_columns)
    print("Categorical columns: ", original_categorical_columns)

    pre_proc_method = "standard"


    # TODO: Should this also return the names of the new columns?
    # Does the reverse_transformers() need to know the column names?
    (
        original_input_transformed,
        reordered_dataframe_columns,
        continuous_transformers,
        categorical_transformers,
        num_categories,
        num_continuous,
    ) = gpu_preproc(train_df,
                       original_continuous_columns,
                       original_categorical_columns,
                       pre_proc_method=pre_proc_method)

    # Not sure we can print shape here if the data is huge

    original_input_transformed.visualize(filename='task_graph.svg')
    
    X_train = original_input_transformed
    print(X_train.columns)
    print("Input data shape: ", X_train.shape[1])

    vae, encoder, decoder = train(config, X_train, num_continuous, num_categories)

    load_vae = VAE(encoder, decoder)
    load_vae.load_state_dict(torch.load(config.filepath))

    # TODO: Figure out how to take generated data and
    # 1. Return it to original format.
    # 2. Write it to disk (idk if we want to do many files or one big file).
    syn_generated_data = generate_diff_size(config, load_vae, X_train, train_df,
                                            reordered_dataframe_columns, continuous_transformers, categorical_transformers, size=None)
    return
    syn_generated_data = syn_generated_data.compute()
    syn_generated_data.to_csv(config.syn_data_path, index=False)
    print("output written")
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
