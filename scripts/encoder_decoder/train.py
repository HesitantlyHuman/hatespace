import torch
from autoclip.torch import QuantileClip
from hatespace.datasets import prepare_dataloaders
from hatespace.models.tokenizer import Tokenizer
from hatespace.training.losses import SequenceLoss
from hatespace.training import EncoderDecoderTrainer
from transformers import EncoderDecoderModel, get_scheduler

MAX_SINGLE_BATCH_SIZE = 8

# TODO: Add a cli, so that running the code is even easier
config = {
    "training_steps": 1_000_000,
    "validation_steps": 100_000,
    "max_learning_rate": 1e-4,
    "batch_size": 64,
    "weight_decay": 0.075,
}


print("Loading transformer models...")
tokenizer = Tokenizer("roberta-base", 512)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "roberta-base", "roberta-base"
)
if torch.cuda.is_available():
    DEVICE = "cuda"
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} cuda devices...")
else:
    DEVICE = "cpu"
    print(f"Using cpu...")
model.to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["max_learning_rate"],
    weight_decay=config["weight_decay"],
)
optimizer = QuantileClip.as_optimizer(
    optimizer, quantile=0.5, history_length=1000, global_threshold=False
)
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=config["training_steps"] * 0.3,
    num_training_steps=config["training_steps"],
)
loss_fn = SequenceLoss(ignore_index=tokenizer.pad_token_id)
print("Loading dataset...")
train_loader, val_loader = prepare_dataloaders(
    "cc_news",
    training_batch_size=config["batch_size"],
    validation_batch_size=config["batch_size"],
    num_workers=12,
)

trainer = EncoderDecoderTrainer(
    "checkpoints/encoder_decoder",
    model=model,
    optimizer=optimizer,
    tokenizer=tokenizer,
    learning_rate_scheduler=lr_scheduler,
    loss_function=loss_fn,
    epochs=10,
    minibatch_size=MAX_SINGLE_BATCH_SIZE,
)
trainer.train(
    training_dataloader=train_loader,
    validation_dataloader=val_loader,
    device=DEVICE,
)
