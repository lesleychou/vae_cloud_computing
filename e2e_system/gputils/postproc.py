from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import dask.dataframe as dd
import cudf
import dask_cudf
import cuml
from dask_ml.preprocessing import StandardScaler, DummyEncoder
from cuml.dask.preprocessing import OneHotEncoder as CumlOneHotEncoder
from dask.distributed import Client, get_client
import pandas as pd
import cupy as cp
import os


def gpu_reverse_transformers(
    synthetic_set,
    original_cols,
    reordered_dataframe_columns,
    original_continuous_columns,
    original_categorical_columns,
    num_continuous,
    cont_transformers=None,
    cat_transformers=None,
    pre_proc_method='standard'
):
    # print(cont_transformers)
    # print(cat_transformers)
    # print(date_transformers)
    

    # synthetic_transformed_set = synthetic_set.copy(deep=True)
    if pre_proc_method == 'standard':
        standardizer = cont_transformers['standard']        
        inverted = synthetic_set[original_continuous_columns].map_partitions(
            standardizer.inverse_transform)
                
        inverted.columns = original_continuous_columns

        result = inverted

        # Have to make this a list because to_dask_array does list comparisions internally.
        one_hot_names = list(reordered_dataframe_columns[:-num_continuous])

        one_hot = cat_transformers['one_hot']
        vals = synthetic_set[one_hot_names].to_dask_array()
        inverted_cats = one_hot.inverse_transform(vals)

        result[original_categorical_columns] = inverted_cats

        result = result[original_cols]
    else:
        print(f'Preprocessing method not supported (inverse transform): {pre_proc_method}')

    return result


def generate_large_data(
        config, 
        vae_model,
        X_train, 
        original_cols,
        reordered_dataframe_columns, 
        continuous_transformers, 
        categorical_transformers, 
        original_continuous_columns, 
        original_categorical_columns, 
        num_continuous,
        size=None
    ):
    if not size:
        size = len(X_train)

    num_parts = X_train.npartitions
    # Torch takes a lot of memory, so need to reduce size of batch generated at each time.
    step = size // (num_parts * 5)

    out_path = config.syn_data_save_dir

    file_num = 0

    print('Writing output to files...')
    for i in tqdm(range(0, size, step)):
        # Generate synthetic data with X_train
        chunk_size = min(step, size - i)
        syn_sample = vae_model.generate(chunk_size)
            
        if torch.cuda.is_available():
            syn_sample = cp.fromDlpack(torch.utils.dlpack.to_dlpack(syn_sample))
            syn_sample = cudf.DataFrame(syn_sample)
            syn_sample.columns = reordered_dataframe_columns
            syn_sample = dask_cudf.from_cudf(syn_sample)
        else:
            syn_sample = pd.DataFrame(
                syn_sample.cpu().detach().numpy(),
                columns=reordered_dataframe_columns
            )
            syn_sample = dd.from_pandas(syn_sample)

        output = gpu_reverse_transformers(
            synthetic_set=syn_sample,
            original_cols=original_cols,
            cont_transformers=continuous_transformers,
            cat_transformers=categorical_transformers,
            pre_proc_method=config.pre_proc_method,
            reordered_dataframe_columns=reordered_dataframe_columns,
            original_continuous_columns=original_continuous_columns,
            original_categorical_columns=original_categorical_columns,
            num_continuous=num_continuous
        )

        # TODO: Is there a better way than writing one file at a time?
        out_file = os.path.join(out_path, f'syn_out_{file_num}.csv')
        output.to_csv(
            out_file, 
            index=False,
            single_file=True
        )

        file_num += 1
        syn_sample = None
        output = None