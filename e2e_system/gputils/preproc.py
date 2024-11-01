import tqdm
import torch
import nvtabular as nvt
import merlin
from merlin.schema import Tags
from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader
import dask.dataframe as dd
import cudf
import dask_cudf
from merlin.core.compat import HAS_GPU, cudf, dask_cudf, device_mem_size


def _collate(batch):
    features, labels = batch[0]
    feature_cols = []
    for feature in features:
        feature_cols.append(features[feature])
    feature_tensor = torch.stack(feature_cols, dim=1)
    label_tensor = labels
    return feature_tensor, label_tensor

def create_loader(
        transformed_ds,
        continuous_columns,
        categorical_columns,
        label_columns,
        batch_size=32
):
    """
    Takes NVT dataset returns a PyTorch Dataloader. This should be used
    after all preprocessing has already been done.
    """
    nvt_iter = TorchAsyncItr(
        transformed_ds,
        cats=categorical_columns,
        conts=continuous_columns,
        labels=label_columns,
        batch_size=batch_size
    )
    loader = DLDataLoader(nvt_iter, collate_fn=_collate)
    return loader


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
    ddf = nvt.Dataset(input_file_paths, **kwargs).to_ddf()

    # Drop any unwanted columns.
    ddf = ddf.drop(excluded_cols).compute()

    original_continuous_columns = []
    original_categorical_columns = []
    # Maybe we should emit this as well
    categorical_len_count = 0
    # Preprocessing (mirrors vanilla read_data method).
    columns = list(ddf.columns)
    for col in tqdm.tqdm(columns):
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
    ds = nvt.Dataset(ddf)

    return ds, original_continuous_columns, original_categorical_columns