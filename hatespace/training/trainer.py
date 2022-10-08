from typing import Dict, Callable, Union, Any, Tuple, List

import os
import torch
from torch.amp import autocast
import numpy as np
from tqdm import tqdm
import transformers
from transformers import logging
import warnings
import hatespace
from hatespace.training.utils import absolute_early_stopping

import torch.distributed as dist

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# TODO configure to use pytorch distributed to send training history to a central node
# central node should be the authoritative rank 0 node
# TODO automatically determine the rank if we are training distributed

# TODO set up trainer to generate random experiment names?

# TODO if distributed and not the master node, then set verbose to false

# TODO allow for a None value checkpoint path to disable checkpointing
# TODO if rank != 0, then disable checkpointing, however cannot set path to None
# TODO if rank != 0, then still load from checkpoint

# TODO delete checkpoint files after training is complete

# TODO allow for validation and training hooks or something, so that we can generate some
# reconstruction samples every epoch

# TODO should we allow for checkpoint frequency which is more frequent than each epoch?


class HatespaceTrainer:
    def __init__(
        self,
        experiment_root: str,
        model: hatespace.models.archetypal.TransformerArchetypal,
        tokenizer: transformers.PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_function: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        epochs: int,
        minibatch_size: int = 2,
        checkpoint_name: str = "checkpoint",
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.distributed = isinstance(
            self.model, torch.nn.parallel.DistributedDataParallel
        )
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.rank = dist.get_rank() if self.distributed else 0
        self.scalar = torch.cuda.amp.GradScaler()
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.learning_rate_scheduler = learning_rate_scheduler
        self.loss_function = loss_function
        self.verbose = (self.rank == 0 and verbose) if self.distributed else verbose

        self.checkpoint_location = experiment_root
        checkpoint_name = checkpoint_name.split(".pt")[0]
        self.checkpoint_filename = checkpoint_name + ".pt"

        self.config = {
            "epochs": epochs,
            "minibatch_size": minibatch_size,
        }
        self.state = {"epoch": 0, "training_history": [], "validation_history": []}

        try:
            checkpoint_filepath = os.path.join(
                self.checkpoint_location, self.checkpoint_filename
            )
            self.load_from_checkpoint(checkpoint_filepath)
            if self.verbose:
                print(
                    f"Found existing training checkpoint in directory '{self.checkpoint_location}'. Resuming training..."
                )
        except FileNotFoundError:
            if self.verbose:
                print(
                    f"Starting new training session in directory '{self.checkpoint_location}'..."
                )
        self.to(next(self.model.parameters()).device)
        _old_train_function = self.train

        def train(
            training_dataloader: torch.utils.data.DataLoader,
            validation_dataloader: torch.utils.data.DataLoader,
            device: str = None,
        ):
            return self._cleanup_if_exception(
                _old_train_function(
                    training_dataloader=training_dataloader,
                    validation_dataloader=validation_dataloader,
                    device=device,
                    verbose=self.verbose,
                )
            )

        self.train = train

    def to(self, device: Union[str, torch.device]) -> None:
        self.device = device
        self.model.to(device=device)

    def batch_prediction(
        self, tokens: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        model_outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=attention_mask,
        )
        predicted_sequence_logits, embeddings = (
            model_outputs.logits,
            model_outputs.embeddings,
        )
        del model_outputs

        return predicted_sequence_logits, embeddings

    def calculate_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        with autocast(device_type=self.device.type):
            model_predictions, embeddings = self.batch_prediction(tokens=batch)
            loss = self.loss_function(
                model_predictions,
                batch["input_ids"],
                embeddings,
            )
        return loss

    def tokenize_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        tokenized_batch = self.tokenizer(batch["data"])
        tokenized_batch = {
            key: tensor.to(self.device) for key, tensor in tokenized_batch.items()
        }
        return tokenized_batch

    def prepare_minibatches(
        self, batch: Dict[str, Any], minibatch_size: int = 2
    ) -> List[Dict[str, Any]]:
        minibatches = []
        for i in range(0, len(batch["data"]), minibatch_size):
            minibatch = {
                key: value[i : i + minibatch_size] for key, value in batch.items()
            }
            minibatches.append(minibatch)
        return minibatches

    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Prepare data
        minibatches = self.prepare_minibatches(
            batch=batch, minibatch_size=self.config["minibatch_size"]
        )

        # Accumulate gradients over mini-batches
        loss = torch.Tensor([0.0]).to(self.device)
        minibatch_count = torch.Tensor([len(minibatches)]).to(self.device)

        def training_ministep(minibatch: Dict[str, Any]) -> torch.Tensor:
            self.optimizer.zero_grad()
            minibatch = self.tokenize_batch(minibatch)
            minibatch_loss = self.calculate_loss(
                batch=minibatch
            )  # TODO Do we need to normalize?
            self.scalar.scale(minibatch_loss).backward()
            return minibatch_loss.detach()

        # TODO cleanup this nonsense
        if self.distributed:
            with self.model.no_sync():
                for minibatch in minibatches[:-1]:
                    loss += training_ministep(minibatch)
            loss += training_ministep(minibatches[-1])
        else:
            for minibatch in minibatches:
                loss += training_ministep(minibatch)

        loss /= minibatch_count
        if self.distributed:
            loss_collection_work = dist.all_reduce(
                loss, op=dist.ReduceOp.SUM, async_op=True
            )

        # Update model parameters
        self.scalar.step(self.optimizer)
        self.scalar.update()
        self.learning_rate_scheduler.step()

        if self.distributed:
            loss_collection_work.wait()
            loss /= self.world_size

        return loss.item()

    def validation_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            minibatches = self.prepare_minibatches(batch=batch)
            loss = torch.Tensor([0.0]).to(self.device)
            minibatch_count = torch.Tensor([len(minibatches)]).to(self.device)
            for minibatch in minibatches:
                minibatch = self.tokenize_batch(minibatch)
                minibatch_loss = self.calculate_loss(batch=minibatch)
                loss += minibatch_loss.detach()
            loss /= minibatch_count

        return loss.item()

    def run_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        step_function: Callable[[torch.nn.Module, Dict[str, Any]], torch.Tensor],
        name: str = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        batch_losses = []
        if verbose:
            data_loader = tqdm(data_loader, desc=name)
        for _, batch in enumerate(data_loader):
            loss = step_function(batch)

            # Update metric tracking
            batch_losses.append(loss)
            if verbose:
                data_loader.set_postfix(
                    {"Avg Loss": "{:4.3f}".format(np.mean(batch_losses[-300:]))}
                )
        return torch.mean(torch.Tensor(batch_losses))

    def train(
        self,
        training_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        device: Union[str, torch.device] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        if not device is None:
            self.to(device=device)

        epochs = self.config["epochs"]
        for epoch in range(self.state["epoch"], epochs):
            if verbose:
                print(f"--- Epoch {epoch}/{epochs} ---")
            self.state["epoch"] = epoch
            self.model.train()
            training_loss = self.run_epoch(
                data_loader=training_dataloader,
                step_function=self.training_step,
                name="Training",
                verbose=verbose,
            )
            self.state["training_history"].append(training_loss)
            self.model.eval()
            validation_loss = self.run_epoch(
                data_loader=validation_dataloader,
                step_function=self.validation_step,
                name="Validation",
                verbose=verbose,
            )
            self.state["validation_history"].append(validation_loss)
            self.checkpoint()

            if validation_loss <= min(self.state["validation_history"]):
                if verbose:
                    print("Validation loss improved, saving new best model...")
                best_model_path = os.path.join(
                    self.checkpoint_location, "best_model.pt"
                )
                torch.save(self.model.state_dict(), best_model_path)

            if absolute_early_stopping(self.state["validation_history"]):
                if verbose:
                    print(
                        f"Validation loss has stopped converging. Halting training after epoch {epoch}... "
                    )
                break

        # TODO fix
        if verbose:
            print(f"Finished training. The best model can be found at {None}.")
        return min(self.state["validation_history"])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.learning_rate_scheduler.state_dict(),
            "trainer": self.state,
            "scalar": self.scalar.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict=state_dict["model"])
        self.optimizer.load_state_dict(state_dict=state_dict["optimizer"])
        self.learning_rate_scheduler.load_state_dict(state_dict=state_dict["scheduler"])
        self.scalar.load_state_dict(state_dict=state_dict["scalar"])
        for key in ["epoch", "training_history", "validation_history"]:
            self.state[key] = state_dict["trainer"][key]

    def load_from_checkpoint(self, checkpoint_filepath: str) -> None:
        state_dict = torch.load(checkpoint_filepath)
        self.load_state_dict(state_dict=state_dict)

    def checkpoint(self) -> None:
        if not os.path.exists(self.checkpoint_location):
            os.makedirs(self.checkpoint_location)
        torch.save(
            self.state_dict(),
            os.path.join(self.checkpoint_location, self.checkpoint_filename),
        )

    def _cleanup_if_exception(
        self,
        function: Callable,
    ) -> torch.Tensor:
        try:
            function()
        except Exception as e:
            self.to("cpu")
            del self.model
            del self.optimizer
            del self.learning_rate_scheduler
            raise e
