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
    "training_steps": 100_000,
    "max_learning_rate": 1e-5,
    "batch_size": 16,
    "weight_decay": 0.01,
}

model_name = "roberta-base"


print("Loading transformer models...")
tokenizer = Tokenizer(model_name, 512)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
if torch.cuda.is_available():
    DEVICE = "cuda"
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} cuda devices...")
else:
    DEVICE = "cpu"
    print(f"Using cpu...")
model.to(DEVICE)

print("Loading dataset...")
train_loader, val_loader = prepare_dataloaders(
    "cc_news",
    training_batch_size=config["batch_size"],
    validation_batch_size=config["batch_size"],
    num_workers=12,
)
training_epoch_length = len(train_loader)
num_epochs = config["training_steps"] // training_epoch_length

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
    num_warmup_steps=(num_epochs * training_epoch_length) * 0.1,
    num_training_steps=num_epochs * training_epoch_length,
)
loss_fn = SequenceLoss(ignore_index=tokenizer.pad_token_id)

trainer = EncoderDecoderTrainer(
    "checkpoints/encoder_decoder",
    model=model,
    optimizer=optimizer,
    tokenizer=tokenizer,
    learning_rate_scheduler=lr_scheduler,
    loss_function=loss_fn,
    epochs=num_epochs,
    minibatch_size=MAX_SINGLE_BATCH_SIZE,
)
trainer.train(
    training_dataloader=train_loader,
    validation_dataloader=val_loader,
    device=DEVICE,
)
