from typing import Callable, Optional, Sequence

from torch.utils.data import Dataset, DataLoader, Sampler
from ironmarch.datasets.base.utils import default_collate_without_key


class DataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler[Sequence]] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        # TODO Potentially change this to support ids which are none, and use the tasks
        # instead of checking for None?
        if collate_fn is None:
            if dataset[0]["target"] is None:
                collate_fn = lambda collate_inputs: default_collate_without_key(
                    collate_inputs, "target"
                )
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
