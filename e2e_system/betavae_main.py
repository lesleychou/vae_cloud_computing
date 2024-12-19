# Standard imports
import numpy as np
import pandas as pd
import torch
import math
import seaborn as sns  
import matplotlib.pyplot as plt
from config import Config
import os
from tqdm import tqdm
from DataSynthesizer.DataDescriber import DataDescriber
from collections import defaultdict, deque


# VAE is in other folder as well as opacus adapted library
import sys
sys.path.append("../")

# Opacus support for differential privacy
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE functions
# from VAE import Decoder, Encoder, VAE 
from BetaVAE import Decoder, Encoder, BetaVAE

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


def build_bayesian_network(input_data_path, original_categorical_columns, degree_of_bayesian_network=2, epsilon=2):
        threshold_value = 10

        dict_categorical_columns = {item: True for item in original_categorical_columns}
        categorical_attributes = dict_categorical_columns

        describer = DataDescriber(category_threshold=threshold_value)
        bn_structure = describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data_path, 
                                                                epsilon=epsilon, 
                                                                k=degree_of_bayesian_network,
                                                                attribute_to_is_categorical=categorical_attributes)

        return bn_structure

def calculate_latent_dim_hierarchy(bn_structure):
    # Build a graph of the BN structure
    graph = defaultdict(list)
    inverse_graph = defaultdict(list)
    all_nodes = set()
    
    for parent, children in bn_structure:
        all_nodes.add(parent)
        all_nodes.update(children)
        for child in children:
            graph[parent].append(child)
            inverse_graph[child].append(parent)
    
    # Initialize a queue for BFS and a dictionary to track the layer of each node
    layer = {}
    queue = deque()
    
    # Start with input nodes (nodes that have no parents)
    for node in all_nodes:
        if len(inverse_graph[node]) == 0:  # Input node (no parents)
            layer[node] = 0
            queue.append(node)
    
    # Perform BFS to assign layers
    while queue:
        node = queue.popleft()
        current_layer = layer[node]
        
        for neighbor in graph[node]:
            if neighbor not in layer:  # If the neighbor hasn't been assigned a layer yet
                layer[neighbor] = current_layer + 1
                queue.append(neighbor)
    
    # Count how many nodes exist at each layer
    layer_counts = defaultdict(int)
    for node in layer:
        layer_counts[layer[node]] += 1
    
    # Create the latent_dim_hierarchy list, which represents the size of each layer
    latent_dim_hierarchy = [layer_counts[i] for i in range(max(layer.values()) + 1)]
    
    return latent_dim_hierarchy


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
    encoder = Encoder(X_train.shape[1], latent_dim_hierarchy=[4, 8, 20], hidden_dim=config.hidden_dim)
    decoder = Decoder([4, 8, 20], num_continuous, hidden_dim=config.hidden_dim, num_categories=num_categories)

    vae = BetaVAE(encoder, decoder)

    print(vae)

    if config.differential_privacy == False:
        vae.train(
            dataloader = data_loader,
            num_epochs=config.n_epochs
        )

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

    # bn_structure = build_bayesian_network(config.output_processed_data_path, original_categorical_columns, degree_of_bayesian_network=2, epsilon=2)
    # print("Bayesian network structure: ", bn_structure)

    # hierarchy_indices = calculate_latent_dim_hierarchy(bn_structure)
    # print(hierarchy_indices)

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
                       pre_proc_method=config.pre_proc_method)

    X_train = original_input_transformed
    print("Input data shape: ", X_train.shape)

    vae, encoder, decoder = train(config, X_train, num_continuous, num_categories)

    # load_vae = HierarchicalVAE(encoder, decoder)
    # load_vae.load_state_dict(torch.load(config.filepath))

    syn_generated_data = generate_diff_size(config, vae, X_train, train_df,
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
