import torch
from autoclip.torch import QuantileClip
from hatespace.datasets import prepare_dataloaders
from hatespace.models.tokenizer import Tokenizer
from hatespace.training.losses import SequenceLoss
from hatespace.training import EncoderDecoderTrainer
from transformers import EncoderDecoderModel, get_scheduler

import argparse

MAX_SINGLE_BATCH_SIZE = 8

# Quick cli
args = argparse.ArgumentParser()
args.add_argument("--batch_size", type=int, default=16)
args.add_argument("--max_learning_rate", type=float, default=3e-5)
args.add_argument("--weight_decay", type=float, default=0.01)
args.add_argument("--training_steps", type=int, default=300_000)
args.add_argument("--warmup_proportion", type=int, default=0.1)
args.add_argument("--dataset", type=str, default="cc_news")
args.add_argument("--data_root", type=str, default="data/cc_news")
args.add_argument("--save_path", type=str, default="checkpoints/encoder_decoder")
args.add_argument("--base_model", type=str, default="roberta-base")
args.add_argument("--pretrained_model_path", type=str, default=None)
args.add_argument("--save_every", type=int, default=750)
args.add_argument("--quantile_clip", type=float, default=0.7)
args.add_argument("--minibatch_size", type=int, default=MAX_SINGLE_BATCH_SIZE)
args.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="Name of the experiment. If None, a name will be generated. Otherwise, the trainer will look for existing checkpoints with the same name and continue training from there if they exist.",
)

config = vars(args.parse_args())

model_name = config["base_model"]

print("Loading transformer models...")
tokenizer = Tokenizer(model_name, 512)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
if config["pretrained_model_path"] is not None:
    state_dict = torch.load(config["pretrained_model_path"])
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
if torch.cuda.is_available():
    DEVICE = "cuda"
    # if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} cuda devices...")
else:
    DEVICE = "cpu"
    print(f"Using cpu...")
model.to(DEVICE)

train_loader, val_loader = prepare_dataloaders(
    config["dataset"],
    training_batch_size=config["batch_size"],
    validation_batch_size=config["batch_size"],
    num_workers=12,
    root=config["data_root"],
)
training_epoch_length = len(train_loader)
num_epochs = config["training_steps"] // training_epoch_length

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["max_learning_rate"],
    weight_decay=config["weight_decay"],
)
optimizer = QuantileClip.as_optimizer(
    optimizer,
    quantile=config["quantile_clip"],
    history_length=1000,
    global_threshold=False,
)
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=(num_epochs * training_epoch_length) * config["warmup_proportion"],
    num_training_steps=num_epochs * training_epoch_length,
)
loss_fn = SequenceLoss(ignore_index=tokenizer.pad_token_id)

trainer = EncoderDecoderTrainer(
    experiment_root=config["save_path"],
    experiment_name=config["experiment_name"],
    model=model,
    optimizer=optimizer,
    tokenizer=tokenizer,
    learning_rate_scheduler=lr_scheduler,
    loss_function=loss_fn,
    epochs=num_epochs,
    minibatch_size=config["minibatch_size"],
    validation_minibatch_size=config["minibatch_size"],
    configuration=config,
)

# We may need to load the dataset again if the seed has changed
print("Loading dataset...")
train_loader, val_loader = prepare_dataloaders(
    config["dataset"],
    training_batch_size=trainer.config["batch_size"],
    validation_batch_size=trainer.config["batch_size"],
    num_workers=12,
    root=trainer.config["data_root"],
    verbose=False,
)

trainer.train(
    training_dataloader=train_loader,
    validation_dataloader=val_loader,
    checkpoint_frequency=config["save_every"],
    device=DEVICE,
)
