import dask_cudf
import dask.dataframe as dd
import cudf
import torch
import queue
import math
import threading


class DataFrameIter:
    """Ripped straight from core.merlin.io. Iterates through partitions of dask DataFrame."""
    
    def __init__(self, ddf, columns=None, indices=None, partition_lens=None, scheduler='synchronous'):
        self.indices = indices if isinstance(indices, list) else list(range(ddf.npartitions))
        self.ddf = ddf
        self.columns = columns
        self.partition_lens = partition_lens if partition_lens else [None] * self.ddf.npartitions
        self.length = None
        self.scheduler = scheduler

    def __call__(self, indices):
        """Sets the indices to iterate over. Length will have to be recomputed though."""
        self.indices = indices
        self.length = None

    def __len__(self):
        """Caches length computation. Assumes that underlying dataframe won't change."""
        if self.length:
            return self.length
        
        # Check that every partition has a length.
        part_lens = [self.partition_lens[i] for i in self.indices if self.partition_lens[i] is not None]
        if len(part_lens) == len(self.indices):
            # if len(part_lens) < len(self.indices):
            #     # Use metadata-based partition-size information
            #     # if/when it is available.  Note that this metadata
            #     # will not be correct if rows where added or dropped
            #     # after IO (within Ops).
            self.length = sum(part_lens[i] for i in self.indices)
            return self.length
        # Computing length manually.
        if len(self.indices) < self.ddf.npartitions:
            self.length = len(self.ddf.partitions[self.indices])
            return self.length
        
        self.length = len(self.ddf)
        return self.length

    def __iter__(self):
        # Compute length and partition lengths while iterating.
        length = 0
        for i in self.indices:
            part = self.ddf.partitions[i]
            if self.columns:
                result = part[self.columns].compute(scheduler=self.scheduler)
            else:
                result = part.compute(scheduler=self.scheduler)

            self.partition_lens[i] = len(result)
            length += self.partition_lens[i]
            yield result

            # Is this here to make sure part gets GC'd?
            part = None
            result = None

        self.length = length

    def __getitem__(self, idx):
        part = self.ddf.partitions[idx]
        if self.columns:
            return part[self.columns].compute(scheduler=self.scheduler)
        else:
            return part.compute(scheduler=self.scheduler)
        

class SequentialBatcher(torch.utils.data.IterableDataset):
    """Makes batches of PyTorch tensors (GPU) out of dask_cudf.DataFrame partitions."""

    def __init__(
        self,
        ddf,
        batch_size=1024,
        shuffle=False,
        dtype=None,
        keep_spill=True
    ):
        self.ddf = ddf
        self.shuffle = shuffle
        self.batch_size = batch_size
        # Default should be float32.
        self.dtype = dtype if dtype else torch.get_default_dtype()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.data = DataFrameIter(ddf)
        # Save the original indices of the data.
        self.indices = self.data.indices

        self.keep_spill = keep_spill

    def __len__(self):
        num_batches = math.ceil(len(self.data) / self.batch_size)
        return num_batches

    def __iter__(self):
        self.data(self.indices)
        if self.create_worker_split():
            # Multiprocessing with CUDA is difficult. Disable for now.
            self.device = 'cpu'
        batch_iterator = self.get_batches()
        for batch_group in batch_iterator:
            for batch in batch_group:
                # Need to put in iterable/tuple because vae.train() unpacks output of data loader.
                yield (batch,)

    def create_worker_split(self):
        """Splits underlying dask df across workers (PyTorch DataLoader)."""
        worker_info = torch.utils.data.get_worker_info()
        # Check for multiprocess.
        if worker_info is not None:
            indices = self.indices
            num_workers = worker_info.num_workers
            per_worker = math.ceil(len(indices) / num_workers)
            worker_id = worker_info.id

            start = worker_id * per_worker
            end = start + per_worker

            # Sets DataFrame iter to iterate a slice.
            self.data(indices[start:end])
            return True
        return False

    def get_batches(self):
        """
        A generator that turns each cuDF partition into a list of torch.Tensor batches.
        Assumes that self.data was initialized already.
        """
        spill: torch.Tensor = None
        for chunk in self.data:
            chunk_tensor = self.df_to_tensor(chunk)
            if spill is not None and spill.numel() > 0:
                chunk_tensor = torch.concat([spill, chunk_tensor])
            batches, spill = self.batch_tensors(chunk_tensor)
            if batches:
                yield batches
            chunk = None
            chunk_tensor = None
            batches = None
        # Emit spillover.
        if spill is not None:
            yield [spill]


    def df_to_tensor(self, chunk):
        df_arr = chunk.values
        tensor = torch.as_tensor(df_arr, device=self.device, dtype=self.dtype)
        return tensor
    
    def batch_tensors(self, chunk_tensor):
        """Splits larger tensor into list of batches. Creates some spill if keep_spill = True."""
        batches = list(torch.split(chunk_tensor, split_size_or_sections=self.batch_size))
        spill = None
        if len(batches) > 0:
            if batches[-1].shape[0] < self.batch_size:
                # Have to clone otherwise spill will eat memory (?).
                if self.keep_spill:
                    spill = batches[-1].clone()
                batches = batches[:-1]
        return batches, spill
    

class ThreadedBatcher(SequentialBatcher):
    """Uses threads to prefetch partitions (and convert them to tensors) in the background."""

    def __init__(self, ddf, batch_size=1024, shuffle=False, dtype=None, keep_spill=True, qsize=1):
        super().__init__(ddf, batch_size, shuffle, dtype, keep_spill)
        self.batch_queue = queue.Queue(qsize)
        self.stop_event = threading.Event()
        self.thread = None
        self.batch_group = None

    def __iter__(self):
        if self.create_worker_split():
            self.device = 'cpu'
        # I'm assuming this start stop stuff is for if it gets reinitialized before it
        # finishes elsewhere?
        self.stop()
        if self.stopped:
            self.start()

        t = threading.Thread(target=self.load_batches)
        t.daemon = True
        t.start()
        self.thread = t

        while True:
            batch_group = self.dequeue()
            for batch in batch_group:
                yield (batch,)
            batch_group = None
            if not self.working and self.empty:
                self.thread = None
                self.batch_group = None
                return
            
    def dequeue(self):
        chunks = self.batch_queue.get()
        if isinstance(chunks, Exception):
            self.stop()
            raise chunks
        return chunks

    def enqueue(self, packet):
        while True:
            if self.stopped:
                return True
            try:
                self.batch_queue.put(packet, timeout=1e-6)
                return False
            except queue.Full:
                continue

    def load_batches(self):
        try:
            self.enqueue_batches()
        except Exception as e:  # pylint: disable=broad-except
            self.enqueue(e)

    def enqueue_batches(self):
        """
        A generator that turns each cuDF partition into a list of torch.Tensor batches.
        Assumes that self.data was initialized already.
        """
        for chunk_batch in self.get_batches():
            if self.stopped:
                return
            if len(chunk_batch) > 0:
                # put returns True if buffer is stopped before
                # packet can be put in queue. Keeps us from
                # freezing on a put on a full queue
                if self.enqueue(chunk_batch):
                    return
            # Does this free memory?
            chunk_batch = None

    @property
    def stopped(self):
        return self.stop_event.is_set()
    
    @property
    def working(self):
        if self.thread is not None:
            return self.thread.is_alive()
        return False
    
    @property
    def empty(self):
        return self.batch_queue.empty()

    def stop(self):
        if self.thread is not None:
            if not self.stopped:
                # Stop thread.
                self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.batch_queue.queue.clear()

    def start(self):
        self.stop_event.clear()