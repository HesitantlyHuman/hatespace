import argparse
import torch
import json
from torch.nn.parallel import DistributedDataParallel
from autoclip.torch import QuantileClip
from hatespace.datasets.prepare import prepare_dataloaders
from hatespace.training.utils import set_global_seed
import hatespace
import hatespace.models
import hatespace.training
from transformers import get_scheduler

import os
from argparse import ArgumentParser

MAX_SINGLE_BATCH_SIZE = 4

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
        "-p",
        "--port",
        default="12355",
        type=str,
        help="port to connect to training cluster",
    )
    parser.add_argument(
        "--directory",
        default="checkpoints/archetypal",
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
        default="iron_march",
        type=str,
        help="dataset to train on",
        choices=["iron_march", "cc_news"],
    )
    parser.add_argument(
        "--latent_dim_size",
        default=512,
        type=int,
        help="size of the latent embedding space, also the number of archetypes of the resulting model",
    )
    parser.add_argument(
        "--epochs",
        default=50,
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
        default=1e-5,
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
        "--reconstruction_loss_weight",
        default=1.0,
        type=float,
        help="weight of the reconstruction loss in the multi criterion loss",
    )
    parser.add_argument(
        "--distribution_loss_weight",
        default=1e2,
        type=float,
        help="weight of the distribution loss in the multi criterion loss",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        default=1.0,
        type=float,
        help="alpha parameter for the dirichlet distribution used in the distribution loss",
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
    os.environ["MASTER_PORT"] = args.port
    try_load_experiment_configuration(args)
    torch.multiprocessing.set_start_method("fork")
    torch.multiprocessing.spawn(
        train_with_config_and_cleanup,
        nprocs=args.gpus,
        args=(args,),
    )

    # TODO add a zipping and uploading step here?
    # maybe pass through the experiment name, so that the zip utility has access as well


def try_load_experiment_configuration(args: argparse.Namespace):
    if args.experiment_name is None:
        return
    checkpoint_location = os.path.join(args.directory, args.experiment_name)
    configuration_path = os.path.join(checkpoint_location, "configuration.json")
    if os.path.exists(checkpoint_location):
        if not os.path.exists(configuration_path):
            raise ValueError(
                f"Checkpoint directory {checkpoint_location} already exists, but no configuration file was found."
            )
        with open(configuration_path, "r") as f:
            saved_args = json.load(f)
            for key, value in saved_args.items():
                setattr(args, key, value)


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
    head = hatespace.models.ArchetypalHead(512, 768, training_config.latent_dim_size)
    model = hatespace.models.TransformerArchetypal.from_pretrained(
        "roberta-base", inner_embedder=head, tokenizer=tokenizer
    )
    # encoder_decoder_state_dict = torch.load(
    #     "checkpoints/encoder_decoder/lower_lr_rate/best_model.pth"
    # )
    # encoder_decoder_state_dict = {
    #     k.replace("module.", ""): v for k, v in encoder_decoder_state_dict.items()
    # }
    # model.load_state_dict(encoder_decoder_state_dict, strict=False)
    model.cuda(process_id)
    if not (training_config.nodes == 1 and training_config.gpus == 1):
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

    distribution_loss_fn = hatespace.training.SampledDirichletLoss(
        alpha=training_config.dirichlet_alpha
    ).cuda(process_id)
    reconstruction_loss_fn = hatespace.training.SequenceLoss(
        ignore_index=tokenizer.pad_token_id
    )
    combined_loss_fn = hatespace.training.HatespaceMultiCriterion(
        reconstruction_loss_fn,
        distribution_loss_fn,
        reconstruction_loss_weight=training_config.reconstruction_loss_weight,
        distribution_loss_weight=training_config.distribution_loss_weight,
        return_dict=True,
    )

    # TODO: Fix loading from checkpoint
    trainer = hatespace.training.ArchetypalTrainer(
        training_config.directory,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        learning_rate_scheduler=lr_scheduler,
        loss_function=combined_loss_fn,
        epochs=training_config.epochs,
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


def train_with_config_and_cleanup(id: int, config: argparse.Namespace):
    try:
        train_with_config(id, config)
        torch.distributed.destroy_process_group()
    except Exception as e:
        torch.distributed.destroy_process_group()
        raise e


if __name__ == "__main__":
    launch()
