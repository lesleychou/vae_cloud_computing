from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import dask.dataframe as dd
import cudf
import dask_cudf
import cuml
from cuml.dask.preprocessing import OneHotEncoder
from cuml.preprocessing import StandardScaler
from dask_ml.wrappers import Incremental
import dask
from gputils.dataloaders import *
# from merlin.core.compat import HAS_GPU, cudf, dask_cudf, device_mem_size


def create_loader(
        transformed_ds,
        batch_size=64,
        dtype=None,
        **kwargs
):
    """
    Takes dask_cudf.DataFrame and returns PyTorch Dataloader. Kwargs
    are passed to the DataLoader.
    """
    dataset = ThreadedBatcher(
        ddf=transformed_ds, 
        batch_size=batch_size,
        dtype=dtype
    )
    # Is multiprocess. Have to make sure that these kwargs
    # are passed for DataLoader to return CUDA Tensors.
    is_mp = kwargs.get('num_workers')
    if is_mp and is_mp > 0 and dataset.device == 'cuda':
        # Default to forkserver.
        context = kwargs.get('multiprocessing_context')
        if not context:
            kwargs['multiprocessing_context'] = 'forkserver'
        kwargs['persistent_workers'] = True

    data_loader = DataLoader(
        dataset,
        batch_size=None,
        **kwargs
    )
    return data_loader


def nvt_read_data(
        input_file_paths, 
        excluded_cols=['time'], 
        output_file_path=None,
        dtype=None,
        **kwargs
):
    """
    Takes a list of csv file names and reads them into a dataset.
    """
    if not dtype:
        dtype = 'float32'
        
    # Getting the ddf (Dask Dataframe) to compute statistics.
    dask.config.set({"dataframe.backend": "cudf"})

    ddf = dd.read_csv(
        input_file_paths,
        **kwargs
    ).astype(dtype)

    # Drop any unwanted columns.
    if excluded_cols:
        ddf = ddf.drop(columns=excluded_cols)

    original_continuous_columns = []
    original_categorical_columns = []
    # Maybe we should emit this as well
    categorical_len_count = 0
    # Preprocessing (mirrors vanilla read_data method).
    columns = list(ddf.columns)
    for col in tqdm(columns):
        # Do not process the value
        n_uniques = ddf[col].unique().compute()
        num_cats = len(n_uniques)
        if num_cats <= 10:
            original_categorical_columns.append(col)
            categorical_len_count += num_cats
        else:
            original_continuous_columns.append(col)

    # Should we write to disk like in the original? Idk if we have the
    # disk space for that. Maybe we make that optional.

    return ddf, original_continuous_columns, original_categorical_columns


def fake_score(y1, y2):
    return 0


def gpu_preproc(
        data_supp: dask_cudf.DataFrame,
        original_continuous_columns,
        original_categorical_columns,
        pre_proc_method='standard',
        output_dtype='float32'
):
    # Specify column configurations
    # original_categorical_columns = [
    #     'load-interval'
    # ]
    # original_continuous_columns = ['output-packet-rate']

    categorical_columns = original_categorical_columns.copy()
    continuous_columns = original_continuous_columns.copy()

    # As of coding this, new version of RDT adds in GMM transformer which is what we require, however hyper transformers do not work as individual
    # transformers take a 'columns' argument that can only allow for fitting of one column - so you need to loop over and create one for each column
    # in order to fit the dataset - https://github.com/sdv-dev/RDT/issues/376

    continuous_transformers = {}
    categorical_transformers = {}

    transformed_dataset = data_supp

    # Convert categoricals (assumed to be numeric) to dtype int. Is this to avoid floating pt error?
    categorical_part = transformed_dataset[categorical_columns].astype(int)
    num_categories = categorical_part.nunique().compute()
    num_categories = list(num_categories.values)

    if pre_proc_method == "standard":
        # Fit cuML standard scaler to all continuous columns.
        # This wrapper repeatedly calls partial_fit() on chunks. It requires a scoring function
        # so just pass it a dummy function.
        standardizer = Incremental(
            StandardScaler(),
            shuffle_blocks=False,
            scoring=fake_score
        )
        temp_columns = transformed_dataset[continuous_columns]
        standardizer.fit(temp_columns)
        cont_ddf = temp_columns.map_partitions(standardizer.transform)
        # Have to pass the fitted estimator because Incremental does not have inverse_transform().
        continuous_transformers["standard"] = standardizer.estimator_
        transformed_dataset[continuous_columns] = cont_ddf
        # cont_ddf.columns = continuous_columns
        # for col, temp_col in zip(continuous_columns, temp_columns.columns):
        #     transformed_dataset[col] = temp_columns[temp_col]

    num_continuous = len(continuous_columns)

    # Make one hot columns out of categorical features.
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    temp_columns = transformed_dataset[categorical_columns]
    one_hot_arr = one_hot_encoder.fit_transform(temp_columns)
    categorical_transformers["one_hot"] = one_hot_encoder

    # Get names for one hot categorical features.
    one_hot_names = list(one_hot_encoder.get_feature_names(categorical_columns))

    # Move one hot features to DataFrame so we can reorder with continuous features.
    # Assign columns individually to preserve ddf index alignment.
    one_hot_arr.compute_chunk_sizes()
    for i, col in enumerate(one_hot_names):
        transformed_dataset[col] = one_hot_arr[:, i]


    # We need the dataframe in the correct format i.e. categorical variables first and in the order of
    # num_categories with continuous variables placed after. Need to specify divisions so that concat will work.

    # Apparently referencing a slice doesn't quite acquire the whole computation graph 
    # associated with the underlying data, so you have to ref the transformed data itself.

    # cont_ddf = cont_ddf.set_index(cat_ddf.index, divisions=cat_ddf.divisions)
    # reordered_ddf = dask_cudf.concat([cat_ddf, cont_ddf], axis=1).astype(output_dtype)

    final_col_names = one_hot_names + continuous_columns
    reordered_ddf = transformed_dataset[final_col_names]

    return (
        reordered_ddf,
        reordered_ddf.columns,
        continuous_transformers,
        categorical_transformers,
        num_categories,
        num_continuous,
    )