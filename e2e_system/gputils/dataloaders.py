import dask_cudf
import dask.dataframe as dd
import cudf
import torch
import queue
import math


class DataFrameIter:
    """Ripped straight from core.merlin.io. Iterates through partitions of dask DataFrame."""
    
    def __init__(self, ddf, columns=None, indices=None, partition_lens=None):
        self.indices = indices if isinstance(indices, list) else range(ddf.npartitions)
        self.ddf = ddf
        self.columns = columns
        self.partition_lens = partition_lens
        self.length = None

    def __len__(self):
        """Caches length computation. Assumes that underlying dataframe won't change."""
        if self.length:
            return self.length
        
        if self.partition_lens:
            # Use metadata-based partition-size information
            # if/when it is available.  Note that this metadata
            # will not be correct if rows where added or dropped
            # after IO (within Ops).
            self.length = sum(self.partition_lens[i] for i in self.indices)
            return self.length
        # Computing length manually.
        if len(self.indices) < self.ddf.npartitions:
            self.length = len(self.ddf.partitions[self.indices])
            return self.length
        
        self.length = len(self.ddf)
        return self.length

    def __iter__(self):
        # Compute length and partition lengths while iterating.
        part_lens = [0] * self.ddf.npartitions
        length = 0
        for i in self.indices:
            part = self.ddf.partitions[i]
            if self.columns:
                result = part[self.columns].compute(scheduler="synchronous")
            else:
                result = part.compute(scheduler="synchronous")

            part_lens[i] = len(result)
            length += part_lens[i]

            yield result
        self.partition_lens = part_lens
        self.length = length
        # Is this here to make sure part gets GC'd?
        part = None

    def __getitem__(self, idx):
        part = self.ddf.get_partition(idx)
        if self.columns:
            return part[self.columns].compute(scheduler="synchronous")
        else:
            return part.compute(scheduler="synchronous")


class SequentialBatcher(torch.utils.data.IterableDataset):
    """Makes batches of PyTorch tensors (GPU) out of dask_cudf.DataFrame partitions."""

    def __init__(
        self,
        ddf,
        batch_size=64,
        shuffle=False,
        dtype=None
    ):
        self.ddf = ddf
        self.data = DataFrameIter(ddf)
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Default should be float32.
        self.dtype = dtype if dtype else torch.get_default_dtype()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        num_batches = math.ceil(len(self.data) / self.batch_size)
        return num_batches

    def __iter__(self):
        batch_iterator = self.get_batches()
        for batch_group in batch_iterator:
            for batch in batch_group:
                # Need to put in iterable/tuple because vae.train() unpacks output of data loader.
                yield (batch,)

    def get_batches(self):
        """A generator that turns each cuDF partition into a list of torch.Tensor batches."""
        spill: torch.Tensor = None
        for chunk in self.data:
            chunk_tensor = self.cudf_to_tensor(chunk)
            if spill is not None and spill.numel() > 0:
                chunk_tensor = torch.concat([spill, chunk_tensor])
            batches, spill = self.batch_tensors(chunk_tensor)
            if batches:
                yield batches
        # Emit spillover.
        if spill is not None:
            yield [spill]


    def cudf_to_tensor(self, chunk):
        df_arr = chunk.values
        tensor = torch.as_tensor(df_arr, device=self.device, dtype=self.dtype)
        return tensor
    
    def batch_tensors(self, chunk_tensor):
        """Splits larger tensor into list of batches. Creates some spill."""
        batches = list(torch.split(chunk_tensor, split_size_or_sections=self.batch_size))
        spill = None
        if len(batches) > 0:
            if batches[-1].shape[0] < self.batch_size:
                spill = batches[-1]
                batches = batches[:-1]
        return batches, spill
