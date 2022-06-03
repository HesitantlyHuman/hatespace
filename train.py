import torch
import numpy as np
import geomloss
from tqdm import tqdm
from hatespace.datasets import IronMarch
from hatespace.datasets.base import DataLoader
from hatespace.models.nlp import Tokenizer
from hatespace.models.archetypal import TransformerArchetypal, LinearArchetypal
from hatespace.training.utils import absolute_early_stopping, velocity_early_stopping
from hatespace.training.losses import SampledDirichletLoss, SequenceLoss
from transformers import get_scheduler

# TODO: Add a cli, so that running the code is even easier

config = {
    "epochs": 50,
    "batch_size": 2,
    "latent_dim_size": 16,
    "num_binary_features": 8,
    "num_reg_features": 0,
    "use_context": False,
    "distribution_weight": 1,  # 0.05
    "dirichlet_alpha": 0.5,
    "gaussian_std": 0.1,  # At or below average dist between points
    "reconstruction_weight": 0,  # 0.003
    "softmax": True,
    "binary_class_weight": 1,
    "use_features": True,
    "feature_threshold": 0.5,
    "bias_weight_strength": 1,
    "reg_class_weight": 1,
    "archetypal_weight": 1,
}

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(f"Using {DEVICE}...")

print("Loading dataset...")
dataset = IronMarch("iron_march_201911")

print("Tokenizing dataset...")
tokenizer = Tokenizer("roberta-base", 512)
dataset = dataset.map(tokenizer, batch_size=256)
train, val = dataset.split(validation_proportion=0.1)

train_loader = DataLoader(train, batch_size=config["batch_size"])
val_loader = DataLoader(val, batch_size=config["batch_size"])

print("Loading transformer models...")
inner_embedder = LinearArchetypal(512 * 768, config["latent_dim_size"])
model = TransformerArchetypal(
    model_name_or_path="roberta-base", inner_embedder=inner_embedder
)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_training_steps = config["epochs"] * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps,
)

distribution_loss_fn = SampledDirichletLoss(alpha=config["dirichlet_alpha"]).to(DEVICE)
reconstruction_loss_fn = SequenceLoss()

print("Starting training...")
losses = {"train": [], "validation": []}
for epoch in range(config["epochs"]):
    print(f'--- Epoch {epoch}/{config["epochs"]} ---')

    batch_losses = []
    p_bar = tqdm(train_loader, desc="Training")
    model.train()
    for batch_num, batch in enumerate(p_bar):
        data = batch["data"]["input_ids"].to(DEVICE)
        model_outputs = model(input_ids=data)
        predicted_sequence_logits, embeddings = (
            model_outputs.logits,
            model_outputs.embeddings,
        )
        del model_outputs

        # Calculate loss
        reconstruction_loss = reconstruction_loss_fn(predicted_sequence_logits, data)
        distribution_loss = distribution_loss_fn(embeddings)

        combined_loss = (
            config["reconstruction_weight"] * reconstruction_loss
            + config["distribution_weight"] * distribution_loss
        )

        # Gradient step
        combined_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Update metric tracking
        batch_losses.append(combined_loss.detach().to("cpu"))
        p_bar.set_postfix({"Loss": "{:4.3f}".format(np.mean(batch_losses[-100:]))})

    batch_losses = []
    p_bar = tqdm(val_loader, desc="Validation")
    model.eval()
    for batch_num, batch in enumerate(p_bar):
        data = batch["data"]["input_ids"].to(DEVICE)
        with torch.no_grad():
            model_outputs = model(input_ids=data)
            predicted_sequence_logits, embeddings = (
                model_outputs.logits,
                model_outputs.embeddings,
            )
            del model_outputs

            # Calculate loss
            reconstruction_loss = reconstruction_loss_fn(
                predicted_sequence_logits, data
            )
            distribution_loss = distribution_loss_fn(embeddings)

        combined_loss = (
            config["reconstruction_weight"] * reconstruction_loss
            + config["distribution_weight"] * distribution_loss
        )

        # Update metric tracking
        batch_losses.append(combined_loss.detach().to("cpu"))
        p_bar.set_postfix({"Loss": "{:4.3f}".format(np.mean(batch_losses[-100:]))})

    if absolute_early_stopping(losses["validation"]):
        print("Early stopping triggered...")
        break
    break

test_string = "Testing if generation is functional"
test_tokens = tokenizer(test_string)["input_ids"]
generated = model.generate_from_sequence(test_tokens.to(DEVICE))
print(tokenizer.decode(generated))

test_embeddings = torch.nn.functional.one_hot(
    torch.Tensor[5], num_classes=config["latent_dim_size"]
)
generated = model.generate_from_embeddings(test_embeddings.to(DEVICE))
print(tokenizer.decode(generated))
