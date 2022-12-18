from typing import Dict, Callable, Union, Any, Tuple, List

import os
import torch
from torch.amp import autocast
import numpy as np
from tqdm import tqdm
import transformers
import json
from transformers import logging
import warnings
import hatespace
from hatespace.training.utils import (
    generate_experiment_name,
    split_batch_into_minibatches,
)

import torch.distributed as dist

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

# TODO adapt trainer to accept both tensors and dicts as loss function outputs

# TODO delete checkpoint files after training is complete

# TODO allow for validation and training hooks or something, so that we can generate some
# reconstruction samples every epoch

# TODO should we allow for checkpoint frequency which is more frequent than each epoch?

# TODO should we load configurations? They don't really effect how we train, but maybe they
# can be a way of verifying that someone intends to resume a previous training run?
# Either way, the way we are handling configurations should be cleaned up a bit.
# Saving the training history in a place where it's easy to access would be nice as well.
# The trainer might not be the right place to solve this, since the random seed is important, for example.


class HatespaceTrainer:
    def __init__(
        self,
        experiment_root: str,
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_function: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        epochs: int,
        experiment_name: str = None,
        minibatch_size: int = 8,
        validation_minibatch_size: int = 2,
        verbose: bool = True,
        configuration: Dict[str, Any] = None,
    ) -> None:
        self.model = model
        self.distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.rank = dist.get_rank() if self.distributed else 0
        self.scalar = torch.cuda.amp.GradScaler()
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.learning_rate_scheduler = learning_rate_scheduler
        self.loss_function = loss_function
        self.verbose = verbose

        self.config = {
            "epochs": epochs,
            "minibatch_size": minibatch_size,
            "validation_minibatch_size": validation_minibatch_size,
            "experiment_name": experiment_name,
            "world_size": self.world_size,
            "rank": self.rank,
            "verbose": verbose,
        }
        if configuration is not None:
            self.config.update(configuration)
        self.state = {"epoch": 0, "training_history": [], "validation_history": []}

        if experiment_root is None:
            self.checkpoint_directory = None
        elif experiment_name is None:
            while True:
                experiment_name = generate_experiment_name()
                self.checkpoint_directory = os.path.join(
                    experiment_root, experiment_name
                )
                self.config["experiment_name"] = experiment_name
                if not os.path.exists(self.checkpoint_directory):
                    break
        else:
            self.checkpoint_directory = os.path.join(experiment_root, experiment_name)
        if self.load_from_checkpoint(self.checkpoint_directory):
            self._log(
                f"Found existing training checkpoint in directory '{self.checkpoint_directory}'. Resuming training..."
            )
        else:
            self._log(
                f"No existing training checkpoint found in directory '{self.checkpoint_directory}'. Starting new training..."
            )
            self.save_configuration()

        self.to(next(self.model.parameters()).device)
        self._wrap_train_with_cleanup()

    def batch_prediction(self, tokens: Dict[str, torch.Tensor]) -> Any:
        raise NotImplementedError

    def calculate_loss(
        self,
        tokens: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device: Union[str, torch.device]) -> None:
        device = torch.device(device)
        self.device = device
        self.model.to(device=device)

    def tokenize_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        tokenized_batch = self.tokenizer(batch["data"])
        tokenized_batch = {
            key: tensor.to(self.device) for key, tensor in tokenized_batch.items()
        }
        return tokenized_batch

    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Prepare data
        minibatches = split_batch_into_minibatches(
            batch=batch, minibatch_size=self.config["minibatch_size"]
        )

        # Accumulate gradients over mini-batches
        loss = torch.Tensor([0.0]).to(self.device)
        minibatch_count = torch.Tensor([len(minibatches)]).to(self.device)

        def training_ministep(minibatch: Dict[str, Any]) -> torch.Tensor:
            self.optimizer.zero_grad(set_to_none=True)
            minibatch = self.tokenize_batch(minibatch)
            with autocast(device_type=self.device.type):
                minibatch_loss = self.calculate_loss(
                    tokens=minibatch
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
            minibatches = split_batch_into_minibatches(
                batch=batch, minibatch_size=self.config["validation_minibatch_size"]
            )
            loss = torch.Tensor([0.0]).to(self.device)
            minibatch_count = torch.Tensor([len(minibatches)]).to(self.device)
            for minibatch in minibatches:
                minibatch = self.tokenize_batch(minibatch)
                with autocast(device_type=self.device.type):
                    minibatch_loss = self.calculate_loss(tokens=minibatch)
                loss += minibatch_loss.detach()
            loss /= minibatch_count

            if self.distributed:
                loss_collection_work = dist.all_reduce(
                    loss, op=dist.ReduceOp.SUM, async_op=True
                )
                loss_collection_work.wait()
                loss /= self.world_size

        return loss.item()

    def run_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        step_function: Callable[[torch.nn.Module, Dict[str, Any]], torch.Tensor],
        name: str = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        batch_losses = []
        if verbose and self.rank == 0:
            data_loader = tqdm(data_loader, desc=name)
        for batch in data_loader:
            loss = step_function(batch)

            # Update metric tracking
            batch_losses.append(loss)
            if verbose and self.rank == 0:
                data_loader.set_postfix(
                    {"Avg Loss": "{:4.3f}".format(np.mean(batch_losses[-300:]))}
                )
        return torch.mean(torch.Tensor(batch_losses))

    def train(
        self,
        training_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        device: Union[str, torch.device] = None,
    ) -> torch.Tensor:
        if not device is None:
            self.to(device=device)

        epochs = self.config["epochs"]
        for epoch in range(self.state["epoch"], epochs):
            self._log(f"--- Epoch {epoch}/{epochs} ---")
            self.state["epoch"] = epoch
            self.model.train()
            training_loss = self.run_epoch(
                data_loader=training_dataloader,
                step_function=self.training_step,
                name="Training",
                verbose=self.verbose,
            )
            self.state["training_history"].append(training_loss)
            self.model.eval()
            validation_loss = self.run_epoch(
                data_loader=validation_dataloader,
                step_function=self.validation_step,
                name="Validation",
                verbose=self.verbose,
            )
            self.state["validation_history"].append(validation_loss)
            self.checkpoint(
                best_model=validation_loss <= min(self.state["validation_history"])
            )

        self._log(
            f"Finished training. The best model can be found at {self.checkpoint_directory}."
        )
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
        """Load the state of the trainer from a state dictionary.

        Warning: Because of memory problems, this function will modify the provided
        state dictionary, deleting the entries."""
        self.optimizer.load_state_dict(state_dict=state_dict["optimizer"])
        del state_dict["optimizer"]
        self.learning_rate_scheduler.load_state_dict(state_dict=state_dict["scheduler"])
        del state_dict["scheduler"]
        self.scalar.load_state_dict(state_dict=state_dict["scalar"])
        del state_dict["scalar"]
        self.model.load_state_dict(state_dict=state_dict["model"])
        del state_dict["model"]
        for key in ["epoch", "training_history", "validation_history"]:
            self.state[key] = state_dict["trainer"][key]
        self.state["epoch"] += 1

    def load_from_checkpoint(self, checkpoint_directory: str) -> bool:
        checkpoint_path = os.path.join(checkpoint_directory, "checkpoint.pt")
        configuration_path = os.path.join(checkpoint_directory, "configuration.json")
        if not os.path.exists(checkpoint_path) or not os.path.exists(
            configuration_path
        ):
            return False
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict=state_dict)
        with open(configuration_path, "r") as f:
            configuration = json.load(f)
        self.config = {key: configuration[key] for key in self.config}
        return True

    def checkpoint(self, best_model: bool = False) -> None:
        if (self.checkpoint_directory is None) or (self.distributed and self.rank != 0):
            return
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        torch.save(
            self.state_dict(),
            os.path.join(self.checkpoint_directory, "checkpoint.pt"),
        )
        self.save_training_history()
        if best_model:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoint_directory, "best_model.pt"),
            )

    def save_training_history(self) -> None:
        if (self.checkpoint_directory is None) or (self.distributed and self.rank != 0):
            return
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        with open(
            os.path.join(self.checkpoint_directory, "training_history.json"), "w"
        ) as f:
            json.dump(
                {
                    "training_history": self.state["training_history"],
                    "validation_history": self.state["validation_history"],
                },
                f,
                indent=4,
            )

    def save_configuration(self) -> None:
        if (self.checkpoint_directory is None) or (self.distributed and self.rank != 0):
            return
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        with open(
            os.path.join(self.checkpoint_directory, "configuration.json"), "w"
        ) as f:
            json.dump(self.config, f, indent=4)

    # TODO consider using python logging module
    def _log(self, message: str) -> None:
        if self.verbose and (not self.distributed or self.rank == 0):
            print(message)

    def _cleanup_if_exception(
        self,
        function: Callable,
    ) -> torch.Tensor:
        try:
            return function()
        except Exception as e:
            self.to("cpu")
            del self.model
            del self.optimizer
            del self.learning_rate_scheduler
            raise e

    def _wrap_train_with_cleanup(self):
        _old_train_function = self.train

        def train(
            training_dataloader: torch.utils.data.DataLoader,
            validation_dataloader: torch.utils.data.DataLoader,
            device: str = None,
        ) -> torch.Tensor:
            return self._cleanup_if_exception(
                lambda: _old_train_function(
                    training_dataloader=training_dataloader,
                    validation_dataloader=validation_dataloader,
                    device=device,
                )
            )

        self.train = train
