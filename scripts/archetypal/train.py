import torch
import numpy as np
from tqdm import tqdm
from hatespace.datasets.prepare import prepare_dataloaders
from hatespace.models import Tokenizer, TransformerArchetypal, ArchetypalHead
from hatespace.training import ArchetypalTrainer
from hatespace.training.losses import (
    HatespaceMultiCriterion,
    SampledDirichletLoss,
    SequenceLoss,
)
from transformers import get_scheduler

# TODO: Add a cli, so that running the code is even easier

config = {
    "epochs": 10,
    "max_learning_rate": 1e-3,
    "latent_dim_size": 512,
    "reconstruction_loss_weight": 10,  # Has a value of 3.0 minimum when trained alone
    "distribution_loss_weight": 0.01,  # Has a value of 0.05 minimum when trained alone
    "dirichlet_alpha": 1.0,
    "weight_decay": 0.1,
    # "gaussian_std": 0.1,  # At or below average dist between points
    # "classification_weight": 1,
}

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(f"Using {DEVICE}...")

print("Loading dataset...")
train_loader, val_loader = prepare_dataloaders(
    "ironmarch",
    training_batch_size=64,
    validation_batch_size=64,
    num_workers=4,
)

print("Loading transformer models...")
tokenizer = Tokenizer("roberta-base", 512)
head = ArchetypalHead(512, 768, config["latent_dim_size"])
model = TransformerArchetypal.from_pretrained(
    "roberta-base", inner_embedder=head, tokenizer=tokenizer
)
encoder_decoder_state_dict = torch.load(
    "checkpoints/encoder_decoder/lower_lr_rate/best_model.pth"
)
encoder_decoder_state_dict = {
    k.replace("module.", ""): v for k, v in encoder_decoder_state_dict.items()
}
model.load_state_dict(encoder_decoder_state_dict, strict=False)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=config["max_learning_rate"])
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
    reconstruction_loss=reconstruction_loss_fn,
    reconstruction_loss_weight=config["reconstruction_loss_weight"],
    distribution_loss=distribution_loss_fn,
    distribution_loss_weight=config["distribution_loss_weight"],
)

trainer = ArchetypalTrainer(
    "checkpoints/archetypal",
    model=model,
    optimizer=optimizer,
    learning_rate_scheduler=lr_scheduler,
    loss_function=combined_loss_fn,
    tokenizer=tokenizer,
    epochs=config["epochs"],
    minibatch_size=4,
)
best_loss = trainer.train(train_loader, val_loader, device=DEVICE)
