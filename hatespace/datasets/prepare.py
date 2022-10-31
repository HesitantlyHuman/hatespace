from typing import Tuple
import torch
from hatespace.datasets import IronMarch, DataLoader
from torch.utils.data._utils.collate import default_collate
import os


def cc_collate(data):
    return default_collate([{"data": data_item["text"]} for data_item in data])


def prepare_dataloaders(
    dataset_name: str,
    training_batch_size: int,
    validation_batch_size: int,
    validation_proportion: float = 0.1,
    num_workers: int = 1,
    world_size: int = 1,
    rank: int = 0,
    root: str = None,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    dataset_name = dataset_name.lower()
    if dataset_name in ["cc_news", "ccnews", "news"]:
        dataset_name = "cc_news"
    elif dataset_name in ["ironmarch", "iron_march"]:
        dataset_name = "iron_march"

    if dataset_name == "cc_news":
        collate_fn = cc_collate
    elif dataset_name == "iron_march":
        collate_fn = None

    if dataset_name == "cc_news":
        import datasets
        from datasets import load_dataset

        if not verbose:
            datasets.logging.set_verbosity_error()
            os.environ["DATASETS_VERBOSITY"] = "critical"

        if root is None:
            root = "data/cc_news"
        # TODO why are we taking the train split and then splitting it again?
        # TODO should obey verbose
        dataset = load_dataset("cc_news", cache_dir=root, keep_in_memory=True)["train"]
        dataset = dataset.train_test_split(train_size=1 - validation_proportion)
        train_set, val_set = dataset["train"], dataset["test"]
    elif dataset_name == "iron_march":
        if root is None:
            root = "data/iron_march"
        dataset = IronMarch(root=root, verbose=verbose)[:1000]
        train_set, val_set = dataset.split(validation_proportion=validation_proportion)

    training_args = {
        "batch_size": training_batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    validation_args = {
        "batch_size": validation_batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=world_size, rank=rank
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_set, num_replicas=world_size, rank=rank
        )
        training_args["sampler"] = train_sampler
        validation_args["sampler"] = val_sampler
        training_args["shuffle"] = False
        validation_args["shuffle"] = False

    train_loader = DataLoader(train_set, **training_args)
    val_loader = DataLoader(val_set, **validation_args)

    return (train_loader, val_loader)
