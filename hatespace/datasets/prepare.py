from typing import Tuple
import torch
from hatespace.datasets import IronMarch, DataLoader
from hatespace.models.tokenizer import Tokenizer
from datasets import load_dataset
from torch.utils.data._utils.collate import default_collate


def prepare_iron_march_dataloaders(
    tokenizer: Tokenizer,
    training_batch_size: int,
    validation_batch_size: int,
    validation_proportion: float = 0.1,
    root: str = "data/iron_march_201911",
) -> Tuple[DataLoader, DataLoader]:
    dataset = IronMarch(root)
    dataset = dataset.map(tokenizer, batch_size=1024)
    train_set, val_set = dataset.split(validation_proportion=validation_proportion)
    train_loader = DataLoader(train_set, batch_size=training_batch_size)
    val_loader = DataLoader(val_set, batch_size=validation_batch_size)
    return (train_loader, val_loader)


def prepare_cc_news_dataloaders(
    tokenizer: Tokenizer,
    training_batch_size: int,
    validation_batch_size: int,
    validation_proportion: float = 0.1,
    root: str = "data/cc_news",
) -> Tuple[DataLoader, DataLoader]:
    dataset = load_dataset("cc_news", cache_dir=root, keep_in_memory=True)["train"]
    tokenizer._return_as_list = False

    def tokenize(examples):
        return {
            key: value.numpy() for key, value in tokenizer(examples["text"]).items()
        }

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=[
            "date",
            "description",
            "domain",
            "image_url",
            "text",
            "title",
            "url",
        ],
    )
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
        dataset["train"], batch_size=training_batch_size, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset["test"], batch_size=validation_batch_size, collate_fn=collate_fn
    )

    return (train_loader, val_loader)


def prepare_dataloaders(
    dataset_name: str,
    tokenizer: Tokenizer,
    training_batch_size: int,
    validation_batch_size: int,
    validation_proportion: float = 0.1,
    root: str = None,
) -> Tuple[DataLoader, DataLoader]:
    dataset_name = dataset_name.lower()
    if root is None:
        kwargs = {}
    else:
        kwargs = {"root": root}
    if dataset_name in ["cc_news", "ccnews", "news"]:
        return prepare_cc_news_dataloaders(
            tokenizer=tokenizer,
            training_batch_size=training_batch_size,
            validation_batch_size=validation_batch_size,
            validation_proportion=validation_proportion,
            **kwargs,
        )
    elif dataset_name in ["ironmarch", "iron_march"]:
        return prepare_iron_march_dataloaders(
            tokenizer=tokenizer,
            training_batch_size=training_batch_size,
            validation_batch_size=validation_batch_size,
            validation_proportion=validation_proportion,
            **kwargs,
        )
