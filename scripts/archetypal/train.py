import torch
from autoclip.torch import QuantileClip
from hatespace.datasets import prepare_dataloaders
from hatespace.models.tokenizer import Tokenizer
from hatespace.models.archetypal import TransformerArchetypal, ArchetypalHead
from hatespace.models.base import Embedder
from hatespace.training.losses import (
    SampledDirichletLoss,
    SequenceLoss,
    HatespaceMultiCriterion,
)
from transformers import get_scheduler

from hatespace.training.trainer import HatespaceTrainer

# TODO: Add a cli, so that running the code is even easier

config = {
    "epochs": 10,
    "learning_rate": 1e-3,
    "batch_size": 2,
    "latent_dim_size": 512,
    "reconstruction_loss_weight": 10,  # Has a value of 3.0 minimum when trained alone
    "distribution_loss_weight": 0.01,  # Has a value of 0.05 minimum when trained alone
    "dirichlet_alpha": 1.0,
    # "gaussian_std": 0.1,  # At or below average dist between points
    # "classification_weight": 1,
}

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(f"Using {DEVICE}...")

print("Loading transformer models...")
head = ArchetypalHead(768 * 512, config["latent_dim_size"])
model = TransformerArchetypal("roberta-base", inner_embedder=head)
model.to(DEVICE)
tokenizer = Tokenizer("roberta-base", 512)

print("Loading dataset...")
train_loader, val_loader = prepare_dataloaders(
    "ironmarch",
    tokenizer=tokenizer,
    training_batch_size=config["batch_size"],
    validation_batch_size=config["batch_size"],
)

optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
optimizer = QuantileClip.as_optimizer(
    optimizer, quantile=0.5, history_length=1000, global_threshold=False
)
num_training_steps = config["epochs"] * len(train_loader)
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=num_training_steps * 0.3,
    num_training_steps=num_training_steps,
)

distribution_loss_fn = SampledDirichletLoss(alpha=config["dirichlet_alpha"]).to(DEVICE)
reconstruction_loss_fn = SequenceLoss(ignore_index=tokenizer.pad_token_id)
combined_loss_fn = HatespaceMultiCriterion(
    reconstruction_loss_fn,
    distribution_loss_fn,
    reconstruction_loss_weight=config["reconstruction_loss_weight"],
    distribution_loss_weight=config["distribution_loss_weight"],
)

trainer = HatespaceTrainer(
    "checkpoints/archetypal/test",
    model=model,
    optimizer=optimizer,
    learning_rate_scheduler=lr_scheduler,
    loss_function=combined_loss_fn,
    epochs=config["epochs"],
)
trainer.train(
    training_dataloader=train_loader,
    validation_dataloader=val_loader,
    device=DEVICE,
)
