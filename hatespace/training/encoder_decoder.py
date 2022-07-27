from typing import Dict, Callable, Generator, Union, Any
from hatespace.training.utils import absolute_early_stopping
import torch
from tqdm import tqdm
import numpy as np
import os

from transformers import logging

logging.set_verbosity_error()


class _GeneratorSlice:
    def __init__(self, generator: Generator, n: int) -> None:
        self.generator = generator
        self.n = n
        self._current_n = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_n < self.n:
            self._current_n += 1
            return next(self.generator)
        else:
            raise StopIteration


class EncoderDecoderTrainer:
    def __init__(
        self,
        experiment_root: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        training_steps: int,
        validation_steps: int,
        checkpoint_frequency: int = 500,
        validation_length: int = 50,
        checkpoint_name: str = "checkpoint",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.learning_rate_scheduler = learning_rate_scheduler
        self.config = {
            "training_steps": training_steps,
            "validation_steps": validation_steps,
            "checkpoint_frequency": checkpoint_frequency,
            "validation_length": validation_length,
        }
        self.state = {"epoch": 0, "training_history": [], "validation_history": []}

        self.checkpoint_location = experiment_root
        checkpoint_name = checkpoint_name.split(".pth")[0]
        self.checkpoint_filename = checkpoint_name + ".pth"
        self.loss_function = loss_function

        try:
            checkpoint_filepath = os.path.join(
                self.checkpoint_location, self.checkpoint_filename
            )
            self.load_from_checkpoint(checkpoint_filepath)
            print(
                f"Found existing training checkpoint in directory '{self.checkpoint_location}'. Resuming training..."
            )
        except FileNotFoundError:
            print(
                f"Starting new training session in directory '{self.checkpoint_location}'..."
            )
        self.to(next(self.model.parameters()).device)
        old_train_function = self.train
        self.train = lambda training_dataloader, validation_dataloader, device=None: self._cleanup_if_exception(
            old_train_function(
                training_dataloader=training_dataloader,
                validation_dataloader=validation_dataloader,
                device=device,
            )
        )

    def to(self, device: Union[str, torch.device]) -> None:
        self.device = device
        self.model.to(device=device)

    def batch_prediction(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["data"]["input_ids"].to(self.device)
        attention_mask = batch["data"]["attention_mask"].to(self.device)
        model_outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=attention_mask,
        )
        predicted_sequence_logits = model_outputs.logits
        del model_outputs

        return predicted_sequence_logits

    def calculate_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        input_ids = batch["data"]["input_ids"].to(self.device)
        model_predictions = self.batch_prediction(batch=batch)
        loss = self.loss_function(model_predictions, input_ids)

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = self.calculate_loss(batch=batch)

        # Gradient step
        loss.backward()
        self.optimizer.step()
        self.learning_rate_scheduler.step()
        self.optimizer.zero_grad()

        return loss.detach()

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            loss = self.calculate_loss(batch=batch)

        return loss.detach()

    def run_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        step_function: Callable[
            [torch.nn.Module, Dict[str, torch.Tensor]], torch.Tensor
        ],
        name: str = None,
    ) -> torch.Tensor:
        batch_losses = []
        progress_bar = tqdm(data_loader, desc=name)
        for _, batch in enumerate(progress_bar):
            loss = step_function(batch)

            # Update metric tracking
            batch_losses.append(loss.to("cpu"))
            progress_bar.set_postfix(
                {"Avg Loss": "{:4.3f}".format(np.mean(batch_losses[-300:]))}
            )
        return torch.mean(torch.Tensor(batch_losses))

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

    def train(
        self,
        training_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        device: Union[str, torch.device] = None,
    ) -> torch.Tensor:
        print("Starting training...")
        if not device is None:
            self.to(device=device)

        def create_data_generator(
            data_loader: torch.utils.data.DataLoader, num_steps: int
        ) -> Generator[dict, None, None]:
            n_batches = 0
            while n_batches < num_steps:
                for batch in data_loader:
                    yield batch
                    n_batches += 1
            return None

        training_dataloader = create_data_generator(
            training_dataloader, self.config["training_steps"]
        )
        validation_dataloader = create_data_generator(
            training_dataloader, self.config["validation_steps"]
        )

        epochs = self.config["training_steps"] // self.config["checkpoint_frequency"]

        for epoch in range(self.state["epoch"], epochs):
            print(f"--- Epoch {epoch}/{epochs} ---")
            self.state["epoch"] = epoch
            self.model.train()
            training_segment = _GeneratorSlice(
                training_dataloader, self.config["checkpoint_frequency"]
            )
            training_loss = self.run_epoch(
                training_segment, self.training_step, "Training"
            )
            self.state["training_history"].append(training_loss)
            self.model.eval()
            validation_segment = _GeneratorSlice(
                validation_dataloader, self.config["validation_length"]
            )
            validation_loss = self.run_epoch(
                validation_segment, self.validation_step, "Validation"
            )
            self.state["validation_history"].append(validation_loss)
            self.checkpoint()

            if validation_loss <= min(self.state["validation_history"]):
                print("Saving new best model...")
                best_model_path = os.path.join(
                    self.checkpoint_location, "best_model.pth"
                )
                torch.save(self.model.state_dict(), best_model_path)

            if absolute_early_stopping(self.state["validation_history"]):
                print(
                    f"Validation loss has stopped converging. Halting training after epoch {epoch}... "
                )
                break

        print(f"Finished training. The best model can be found at {None}.")
        return min(self.state["validation_history"])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.learning_rate_scheduler.state_dict(),
            "trainer": self.state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict=state_dict["model"])
        self.optimizer.load_state_dict(state_dict=state_dict["optimizer"])
        self.learning_rate_scheduler.load_state_dict(state_dict=state_dict["scheduler"])
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
