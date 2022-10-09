import argparse
import torch
from torch.nn.parallel import DistributedDataParallel
from autoclip.torch import QuantileClip
from hatespace.datasets.prepare import prepare_distributed_dataloaders
import hatespace
import hatespace.models
import hatespace.training
from transformers import get_scheduler

import os
from argparse import ArgumentParser

MAX_SINGLE_BATCH_SIZE = 4

default_config = {
    "epochs": 10,
    "learning_rate": 1e-3,
    "latent_dim_size": 512,
    "reconstruction_loss_weight": 10,  # Has a value of 3.0 minimum when trained alone
    "distribution_loss_weight": 0.01,  # Has a value of 0.05 minimum when trained alone
    "dirichlet_alpha": 1.0,
    "weight_decay": 0.1,
    # "gaussian_std": 0.1,  # At or below average dist between points
    # "classification_weight": 1,
}

# TODO do a final dry run with a slice of the dataset

# Figure out how we're going to save configurations
# should we use WandB?


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
        "--epochs",
        default=10,
        type=int,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="batch size to train with",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="number of workers to use per GPU for data loading",
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
        args=(args, default_config),
    )

    # TODO add a zipping and uploading step here?
    # maybe pass through the experiment name, so that the zip utility has access as well


def train_with_config(
    process_id: int, training_config: argparse.Namespace, hyperparameters: dict
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
    torch.manual_seed(0)

    if process_id == 0:
        print("Loading transformer models...")
    tokenizer = hatespace.models.Tokenizer("roberta-base", max_length=512)
    head = hatespace.models.ArchetypalHead(512, 768, hyperparameters["latent_dim_size"])
    model = hatespace.models.TransformerArchetypal("roberta-base", head)
    model = DistributedDataParallel(
        model, device_ids=[process_id], find_unused_parameters=True
    )

    train_loader, val_loader = prepare_distributed_dataloaders(
        "ironmarch",
        training_batch_size=training_config.per_gpu_batch_size,
        validation_batch_size=training_config.per_gpu_batch_size,
        num_workers=training_config.num_workers,
        world_size=training_config.world_size,
        rank=rank,
        verbose=process_id == 0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters["learning_rate"],
        weight_decay=hyperparameters["weight_decay"],
    )
    optimizer = QuantileClip.as_optimizer(
        optimizer, quantile=0.8, history_length=1000, global_threshold=False
    )
    num_training_steps = hyperparameters["epochs"] * len(train_loader)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps * 0.3,
        num_training_steps=num_training_steps,
    )

    distribution_loss_fn = hatespace.training.SampledDirichletLoss(
        alpha=hyperparameters["dirichlet_alpha"]
    ).cuda(process_id)
    reconstruction_loss_fn = hatespace.training.SequenceLoss(
        ignore_index=tokenizer.pad_token_id
    )
    combined_loss_fn = hatespace.training.HatespaceMultiCriterion(
        reconstruction_loss_fn,
        distribution_loss_fn,
        reconstruction_loss_weight=hyperparameters["reconstruction_loss_weight"],
        distribution_loss_weight=hyperparameters["distribution_loss_weight"],
    )

    # TODO: Configure the checkpointing to work with a remote server
    # ^ we won't worry about this, because we can just zip the experiment folder
    # afterwards and upload it to the server
    # TODO: Ensure that only the master process saves checkpoints
    # TODO: Fix loading from checkpoint
    trainer = hatespace.training.HatespaceTrainer(
        "checkpoints/bleh",
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        learning_rate_scheduler=lr_scheduler,
        loss_function=combined_loss_fn,
        epochs=hyperparameters["epochs"],
        minibatch_size=MAX_SINGLE_BATCH_SIZE,
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
