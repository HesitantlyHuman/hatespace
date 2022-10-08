from typing import Tuple
import torch
from hatespace.datasets import IronMarch, DataLoader
from hatespace.models.tokenizer import Tokenizer
from datasets import load_dataset
from torch.utils.data._utils.collate import default_collate


def prepare_iron_march_dataloaders(
    training_batch_size: int,
    validation_batch_size: int,
    validation_proportion: float = 0.1,
    num_workers: int = 1,
    root: str = "data/iron_march_201911",
) -> Tuple[DataLoader, DataLoader]:
    dataset = IronMarch(root)
    train_set, val_set = dataset.split(validation_proportion=validation_proportion)
    train_loader = DataLoader(
        train_set,
        batch_size=training_batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=validation_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return (train_loader, val_loader)


# Probably better to refactor the original prepare function
# but this was a quicker way to get something working
def prepare_distributed_iron_march_dataloaders(
    training_batch_size: int,
    validation_batch_size: int,
    validation_proportion: float = 0.1,
    num_workers: int = 1,
    world_size: int = 1,
    rank: int = 0,
    root: str = "data/iron_march_201911",
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    dataset = IronMarch(root, verbose=verbose)
    train_set, val_set = dataset.split(validation_proportion=validation_proportion)
    train_loader = DataLoader(
        train_set,
        batch_size=training_batch_size,
        shuffle=False,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=world_size, rank=rank
        ),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=validation_batch_size,
        shuffle=False,
        sampler=torch.utils.data.distributed.DistributedSampler(
            val_set, num_replicas=world_size, rank=rank
        ),
        num_workers=num_workers,
        pin_memory=True,
    )
    return (train_loader, val_loader)


def prepare_cc_news_dataloaders(
    training_batch_size: int,
    validation_batch_size: int,
    validation_proportion: float = 0.1,
    num_workers: int = 1,
    root: str = "data/cc_news",
) -> Tuple[DataLoader, DataLoader]:
    dataset = load_dataset("cc_news", cache_dir=root, keep_in_memory=True)["train"]
    dataset = dataset.train_test_split(train_size=1 - validation_proportion)

    def collate_fn(data):
        return default_collate(
            [
                {
                    "data": {
                        key: torch.LongTensor(value) for key, value in data_item.items()
                    }
                }
                for data_item in data
            ]
        )

    train_loader = DataLoader(
        dataset["train"],
        batch_size=training_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset["test"],
        batch_size=validation_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (train_loader, val_loader)


def prepare_distributed_cc_news_dataloaders(
    training_batch_size: int,
    validation_batch_size: int,
    validation_proportion: float = 0.1,
    num_workers: int = 1,
    world_size: int = 1,
    rank: int = 0,
    root: str = "data/cc_news",
) -> Tuple[DataLoader, DataLoader]:
    dataset = load_dataset("cc_news", cache_dir=root, keep_in_memory=True)["train"]
    dataset = dataset.train_test_split(train_size=1 - validation_proportion)

    def collate_fn(data):
        return default_collate(
            [
                {
                    "data": {
                        key: torch.LongTensor(value) for key, value in data_item.items()
                    }
                }
                for data_item in data
            ]
        )

    train_loader = DataLoader(
        dataset["train"],
        batch_size=training_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset["train"], num_replicas=world_size, rank=rank
        ),
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset["test"],
        batch_size=validation_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset["test"], num_replicas=world_size, rank=rank
        ),
        pin_memory=True,
    )

    return (train_loader, val_loader)


def prepare_dataloaders(
    dataset_name: str,
    training_batch_size: int,
    validation_batch_size: int,
    validation_proportion: float = 0.1,
    num_workers: int = 1,
    root: str = None,
) -> Tuple[DataLoader, DataLoader]:
    dataset_name = dataset_name.lower()
    if root is None:
        kwargs = {}
    else:
        kwargs = {"root": root}
    if dataset_name in ["cc_news", "ccnews", "news"]:
        return prepare_cc_news_dataloaders(
            training_batch_size=training_batch_size,
            validation_batch_size=validation_batch_size,
            validation_proportion=validation_proportion,
            num_workers=num_workers,
            **kwargs,
        )
    elif dataset_name in ["ironmarch", "iron_march"]:
        return prepare_iron_march_dataloaders(
            training_batch_size=training_batch_size,
            validation_batch_size=validation_batch_size,
            validation_proportion=validation_proportion,
            num_workers=num_workers,
            **kwargs,
        )


def prepare_distributed_dataloaders(
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
    if root is None:
        kwargs = {}
    else:
        kwargs = {"root": root}
    if dataset_name in ["cc_news", "ccnews", "news"]:
        return prepare_distributed_cc_news_dataloaders(
            training_batch_size=training_batch_size,
            validation_batch_size=validation_batch_size,
            validation_proportion=validation_proportion,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            verbose=verbose,
            **kwargs,
        )
    elif dataset_name in ["ironmarch", "iron_march"]:
        return prepare_distributed_iron_march_dataloaders(
            training_batch_size=training_batch_size,
            validation_batch_size=validation_batch_size,
            validation_proportion=validation_proportion,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            verbose=verbose,
            **kwargs,
        )
