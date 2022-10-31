import argparse
import torch
from torch.nn.parallel import DistributedDataParallel
from autoclip.torch import QuantileClip
from hatespace.datasets.prepare import prepare_dataloaders
import hatespace
import hatespace.training
from transformers import EncoderDecoderModel, get_scheduler
import random
import os
from argparse import ArgumentParser

from hatespace.training.utils import set_global_seed

MAX_SINGLE_BATCH_SIZE = 8

default_config = {
    "epochs": 10,
    "max_learning_rate": 1e-3,
    "latent_dim_size": 512,
    "weight_decay": 0.1,
}

# Figure out how we're going to save configurations
# should we use WandB?

# TODO allow for selecting specific gpus to use

# TODO add host and port options


def launch():
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--nodes",
        default=1,
        type=int,
        help="number of nodes to train with",
    )
    parser.add_argument(
        "-nr",
        "--node_rank",
        default=0,
        type=int,
        help="specify which node is being launched",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        default=1,
        type=int,
        help="number of gpus per node to train with",
    )
    parser.add_argument(
        "--directory",
        default="checkpoints/encoder_decoder",
        type=str,
        help="directory to save checkpoints to",
    )
    parser.add_argument(
        "--experiment_name",
        default=None,
        type=str,
        help="name of the experiment",
    )
    parser.add_argument(
        "--dataset",
        default="cc_news",
        type=str,
        help="dataset to train on",
        choices=["iron_march", "cc_news"],
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="batch size to train with",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="number of workers to use per GPU for data loading",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="the maximum learning rate to hit during the one cycle policy",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float,
        help="weight decay to use for the optimizer",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="seed for any random processes",
    )
    args = parser.parse_args()
    args.per_gpu_batch_size = args.batch_size // args.gpus
    args.world_size = args.gpus * args.nodes
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.multiprocessing.set_start_method("fork")
    torch.multiprocessing.spawn(
        train_with_config,
        nprocs=args.gpus,
        args=(args,),
    )

    # TODO add a zipping and uploading step here?
    # maybe pass through the experiment name, so that the zip utility has access as well


def train_with_config(
    process_id: int,
    training_config: argparse.Namespace,
):
    rank = training_config.node_rank * training_config.gpus + process_id
    print(f"Initalizing node process group {process_id} with rank {rank}...")
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=training_config.world_size,
        rank=rank,
    )
    torch.cuda.set_device(process_id)
    set_global_seed(training_config.seed)

    if process_id == 0:
        print("Loading transformer models...")
    tokenizer = hatespace.models.Tokenizer("roberta-base", max_length=512)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        "roberta-base", "roberta-base"
    )
    model.cuda(process_id)
    model = DistributedDataParallel(
        model, device_ids=[process_id], find_unused_parameters=True
    )

    train_loader, val_loader = prepare_dataloaders(
        training_config.dataset,
        training_batch_size=training_config.per_gpu_batch_size,
        validation_batch_size=training_config.per_gpu_batch_size,
        num_workers=training_config.num_workers,
        world_size=training_config.world_size,
        rank=rank,
        verbose=process_id == 0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    optimizer = QuantileClip.as_optimizer(
        optimizer, quantile=0.8, history_length=1000, global_threshold=False
    )
    num_training_steps = training_config.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps * 0.3,
        num_training_steps=num_training_steps,
    )
    sequence_loss = hatespace.training.SequenceLoss(ignore_index=tokenizer.pad_token_id)

    # TODO: Fix loading from checkpoint RAM error
    trainer = hatespace.training.EncoderDecoderTrainer(
        training_config.directory,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        learning_rate_scheduler=lr_scheduler,
        loss_function=sequence_loss,
        epochs=training_config.epochs,
        minibatch_size=MAX_SINGLE_BATCH_SIZE,
        configuration=vars(training_config),
        experiment_name=training_config.experiment_name,
    )
    best_loss = trainer.train(
        training_dataloader=train_loader,
        validation_dataloader=val_loader,
        device=torch.device(f"cuda:{process_id}"),
    )
    print(f"--- Finished training ---")
    print(f"Best loss: {best_loss}")

    # TODO: Do this if exception is thrown?
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    launch()
