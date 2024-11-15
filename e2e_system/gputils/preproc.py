from tqdm import tqdm
import torch
# import nvtabular as nvt
# import merlin
# from merlin.schema import Tags
# from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader
import dask.dataframe as dd
import cudf
import dask_cudf
import cuml
from cuml.dask.preprocessing import OneHotEncoder
from cuml.preprocessing import StandardScaler
from dask_ml.wrappers import Incremental
# from merlin.core.compat import HAS_GPU, cudf, dask_cudf, device_mem_size


def _collate(batch):
    features, labels = batch[0]
    feature_cols = []
    for feature in features:
        feature_cols.append(features[feature])
    feature_tensor = torch.stack(feature_cols, dim=1)
    label_tensor = labels
    return feature_tensor, label_tensor

# def create_loader(
#         transformed_ds,
#         continuous_columns,
#         categorical_columns,
#         label_columns,
#         batch_size=32
# ):
#     """
#     Takes NVT dataset returns a PyTorch Dataloader. This should be used
#     after all preprocessing has already been done.
#     """
#     nvt_iter = TorchAsyncItr(
#         transformed_ds,
#         cats=categorical_columns,
#         conts=continuous_columns,
#         labels=label_columns,
#         batch_size=batch_size
#     )
#     loader = DLDataLoader(nvt_iter, collate_fn=_collate)
#     return loader


def nvt_read_data(
        input_file_paths, 
        excluded_cols=['time'], 
        output_file_path=None, 
        **kwargs
):
    """
    Takes a list of csv file names and reads them into a dataset.
    """
    # Getting the ddf (Dask Dataframe) to compute statistics.
    blocksize = int(1e+9)
    ddf = dask_cudf.read_csv(
        input_file_paths,
        blocksize=blocksize
    )

    # Drop any unwanted columns.
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


def gpu_preproc(
        data_supp: dask_cudf.DataFrame,
        original_continuous_columns,
        original_categorical_columns,
        pre_proc_method="standard"
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


    # Convert categoricals (assumed to be numeric) to dtype int.
    categorical_part = transformed_dataset[categorical_columns].astype(int)
    num_categories = categorical_part.nunique().compute()
    num_categories = list(num_categories.values)

    if pre_proc_method == "standard":
        # Fit cuML standard scaler to all continuous columns.
        print(transformed_dataset.head())
        # This wrapper repeatedly calls partial_fit() on chunks. It requires a scoring function
        # so just pass it a dummy function.
        temp_continuous = Incremental(
            StandardScaler(),
            shuffle_blocks=False,
            scoring=lambda y1, y2: 0
        )
        temp_columns = transformed_dataset[continuous_columns]
        temp_continuous.fit(temp_columns)
        temp_columns = temp_columns.map_partitions(temp_continuous.transform)
        continuous_transformers["continuous_"] = temp_continuous
        transformed_dataset[continuous_columns] = temp_columns
        # for col, temp_col in zip(continuous_columns, temp_columns.columns):
        #     transformed_dataset[col] = temp_columns[temp_col]

        print(transformed_dataset.head())

    num_continuous = len(continuous_columns)

    temp_categorical = OneHotEncoder()
    temp_columns = transformed_dataset[categorical_columns]
    transformed_cols = temp_categorical.fit_transform(temp_columns)
    categorical_transformers[
            "categorical_"
    ] = temp_categorical

    print(num_categories)

    # Get names for one hot categorical features.
    one_hot_names = list(temp_categorical.get_feature_names())
    print(one_hot_names)
    print(transformed_cols)

    # Add one hot features.
    transformed_cols.compute_chunk_sizes()
    temp_ddf = transformed_cols.to_dask_dataframe()

    transformed_dataset[one_hot_names] = temp_ddf

    print(transformed_dataset.head())


    # We need the dataframe in the correct format i.e. categorical variables first and in the order of
    # num_categories with continuous variables placed after

    return

    reordered_dataframe = transformed_dataset.iloc[:, num_continuous:]

    reordered_dataframe = pd.concat(
        [reordered_dataframe, transformed_dataset.iloc[:, :num_continuous]],
        axis=1,
    )

    x_train_df = reordered_dataframe.astype('float32')

    return (
        x_train_df,
        reordered_dataframe.columns,
        continuous_transformers,
        categorical_transformers,
        num_categories,
        num_continuous,
    )