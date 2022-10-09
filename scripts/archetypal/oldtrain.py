import torch
import numpy as np
from tqdm import tqdm
from hatespace.datasets import IronMarch, DataLoader
from hatespace.models import Tokenizer
from hatespace.models.model import TransformerArchetypal, LinearArchetypal
from hatespace.training.utils import absolute_early_stopping, velocity_early_stopping
from hatespace.training.losses import SampledDirichletLoss, SequenceLoss
from transformers import (
    EncoderDecoderModel,
    get_scheduler,
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaModel,
    RobertaForCausalLM,
)

# TODO: Add a cli, so that running the code is even easier

config = {
    "epochs": 10,
    "learning_rate": 1e-3,
    "batch_size": 4,
    "latent_dim_size": 512,
    "max_grad_norm": 1.0,
    # "use_features": True,
    # "num_binary_features": 8,
    # "binary_feature_threshold": 0.5,
    # "num_reg_features": 0,
    # "use_context": False,
    "distribution_weight": 0.01,  # 0.05
    "dirichlet_alpha": 1.0,
    # "gaussian_std": 0.1,  # At or below average dist between points
    "reconstruction_weight": 10,  # 3.0
    # "binary_class_weight": 1,
    # "bias_weight_strength": 1,
    # "reg_class_weight": 1,
    # "archetypal_weight": 1,
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
# tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


# def tokenize_function(inputs):
#     outputs = tokenizer(inputs, max_length=512, padding="max_length")
#     keys = outputs.keys()
#     return [
#         {key: torch.Tensor(value[:512]).long() for key, value in zip(keys, value_tuple)}
#         for value_tuple in zip(*outputs.values())
#     ]


dataset = dataset.map(tokenizer, batch_size=256)
train, val = dataset.split(validation_proportion=0.1)
test_string = "Hehe, this is a test string which we want to reconstruct. Yay!"
test_tokens = tokenizer([test_string])
test_tokens = {
    key: torch.Tensor(value).long().to(DEVICE) for key, value in test_tokens.items()
}

train_loader = DataLoader(train, batch_size=config["batch_size"])
val_loader = DataLoader(val, batch_size=config["batch_size"])

print("Loading transformer models...")
# inner_embedder = LinearArchetypal(512 * 768, config["latent_dim_size"])
# model = TransformerArchetypal(
#     model_name_or_path="roberta-base", inner_embedder=inner_embedder
# )

# Define encoder decoder model
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "roberta-base", "roberta-base"
)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
num_training_steps = config["epochs"] * len(train_loader)
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=num_training_steps * 0.3,
    num_training_steps=num_training_steps,
)

# distribution_loss_fn = SampledDirichletLoss(alpha=config["dirichlet_alpha"]).to(DEVICE)
reconstruction_loss_fn = SequenceLoss(ignore_index=tokenizer.pad_token_id)

with torch.no_grad():
    output_tokens = model.generate(**test_tokens)
output_string = tokenizer.batch_decode(output_tokens)
print(f"Initial reconstruction:\n{output_string}")

print("Starting training...")
losses = {"train": [], "validation": []}
for epoch in range(config["epochs"]):
    print(f'--- Epoch {epoch}/{config["epochs"]} ---')

    batch_losses = []
    p_bar = tqdm(train_loader, desc="Training")
    model.train()
    for batch_num, batch in enumerate(p_bar):
        input_ids = batch["data"]["input_ids"].to(DEVICE)
        attention_mask = batch["data"]["attention_mask"].to(DEVICE)
        model_outputs = model(
            input_ids=input_ids,
            decoder_input_ids=input_ids,
            # attention_mask=attention_mask,
            # decoder_attention_mask=attention_mask,
        )
        predicted_sequence_logits, embeddings = (
            model_outputs.logits,
            None,  # model_outputs.embeddings,
        )
        del model_outputs

        # Calculate loss
        reconstruction_loss = reconstruction_loss_fn(
            predicted_sequence_logits, input_ids
        )
        # distribution_loss = distribution_loss_fn(embeddings)

        combined_loss = reconstruction_loss
        # combined_loss = (
        #     config["reconstruction_weight"] * reconstruction_loss
        #     + config["distribution_weight"] * distribution_loss
        # )

        # Gradient step
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["max_grad_norm"]
        )  # TODO kinda sus bro
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
        input_ids = batch["data"]["input_ids"].to(DEVICE)
        attention_mask = batch["data"]["attention_mask"].to(DEVICE)
        with torch.no_grad():
            model_outputs = model(
                input_ids=input_ids,
                decoder_input_ids=input_ids,
                # attention_mask=attention_mask,
                # decoder_attention_mask=attention_mask,
            )
            predicted_sequence_logits, embeddings = (
                model_outputs.logits,
                None,  # model_outputs.embeddings,
            )
            del model_outputs

            # Calculate loss
            reconstruction_loss = reconstruction_loss_fn(
                predicted_sequence_logits, input_ids
            )
            # distribution_loss = distribution_loss_fn(embeddings)

        combined_loss = reconstruction_loss
        # combined_loss = (
        #     config["reconstruction_weight"] * reconstruction_loss
        #     + config["distribution_weight"] * distribution_loss
        # )

        # Update metric tracking
        batch_losses.append(combined_loss.detach().to("cpu"))
        p_bar.set_postfix({"Loss": "{:4.3f}".format(np.mean(batch_losses[-100:]))})

    with torch.no_grad():
        output_tokens = model.generate(**test_tokens)
    output_string = tokenizer.batch_decode(output_tokens)
    print(f"Intermediate reconstruction:\n{output_string}")

# test_embeddings = torch.nn.functional.one_hot(
#     torch.Tensor[5], num_classes=config["latent_dim_size"]
# )
# generated = model.generate_from_embeddings(test_embeddings.to(DEVICE))
# print(tokenizer.decode(generated))
