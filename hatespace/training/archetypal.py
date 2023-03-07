from typing import Dict, Tuple, Any, Callable, Union

import torch
from torch.amp import autocast
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from hatespace.training.trainer import HatespaceTrainer
from hatespace.training.trainer import split_batch_into_minibatches
from hatespace.training.utils import report_cuda_memory_info


class ArchetypalTrainer(HatespaceTrainer):
    def __init__(
        self,
        experiment_root: str,
        model: torch.nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_function: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        epochs: int,
        experiment_name: str = None,
        minibatch_size: int = 8,
        validation_minibatch_size: int = 2,
        verbose: bool = True,
        configuration: Dict[str, Any] = None,
    ) -> None:

        super().__init__(
            experiment_root=experiment_root,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            learning_rate_scheduler=learning_rate_scheduler,
            loss_function=loss_function,
            epochs=epochs,
            experiment_name=experiment_name,
            minibatch_size=minibatch_size,
            validation_minibatch_size=validation_minibatch_size,
            verbose=verbose,
            configuration=configuration,
        )
        if self.state["epoch"] == 0:
            self.state = {
                "epoch": 0,
                "training_history": {
                    "loss": [],
                    "reconstruction_loss": [],
                    "distribution_loss": [],
                },
                "validation_history": {
                    "loss": [],
                    "reconstruction_loss": [],
                    "distribution_loss": [],
                },
            }

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
        predicted_sequence_logits, embeddings, feature_predictions = (
            model_outputs.logits,
            model_outputs.embeddings,
            model_outputs.feature_predictions,
        )
        del model_outputs

        return predicted_sequence_logits, embeddings, feature_predictions

    def calculate_loss(
        self,
        tokens: Dict[str, torch.Tensor],
        features: torch.Tensor # TODO: determine the actual type. torch.Tensor is placeholder.
    ) -> torch.Tensor:
        model_predictions, embeddings, feature_predictions = self.batch_prediction(tokens=tokens)
        loss = self.loss_function(
            model_predictions,
            tokens["input_ids"],
            embeddings,
            feature_predictions,
            features
        )
        return loss

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Prepare data
        minibatches = split_batch_into_minibatches(
            batch=batch, minibatch_size=self.config["minibatch_size"]
        )

        # Accumulate gradients over mini-batches
        loss = torch.Tensor([0.0]).to(self.device)
        reconstruction_loss = torch.Tensor([0.0]).to(self.device)
        distribution_loss = torch.Tensor([0.0]).to(self.device)
        side_info_loss = torch.Tensor([0.0]).to(self.device)
        minibatch_count = torch.Tensor([len(minibatches)]).to(self.device)

        def training_ministep(minibatch: Dict[str, Any]) -> torch.Tensor:
            self.optimizer.zero_grad(set_to_none=True)
            minibatch = self.tokenize_batch(minibatch)
            with autocast(device_type=self.device.type):
                minibatch_losses = self.calculate_loss(
                    tokens=minibatch
                )  # TODO Do we need to normalize?
            self.scalar.scale(minibatch_losses["loss"]).backward()
            minibatch_losses["loss"] = minibatch_losses["loss"].detach()
            return minibatch_losses

        # TODO cleanup this nonsense
        if self.distributed:
            with self.model.no_sync():
                for minibatch in minibatches[:-1]:
                    minibatch_losses = training_ministep(minibatch)
                    loss += minibatch_losses["loss"]
                    reconstruction_loss += minibatch_losses["reconstruction_loss"]
                    distribution_loss += minibatch_losses["distribution_loss"]
                    side_info_loss += minibatch_losses["side_info_loss"]
            minibatch_losses = training_ministep(minibatches[-1])
            loss += minibatch_losses["loss"]
            reconstruction_loss += minibatch_losses["reconstruction_loss"]
            distribution_loss += minibatch_losses["distribution_loss"]
            side_info_loss += minibatch_losses["side_info_loss"]
        else:
            for minibatch in minibatches:
                minibatch_losses = training_ministep(minibatch)
                loss += minibatch_losses["loss"]
                reconstruction_loss += minibatch_losses["reconstruction_loss"]
                distribution_loss += minibatch_losses["distribution_loss"]
                side_info_loss += minibatch_losses["side_info_loss"]

        loss /= minibatch_count
        reconstruction_loss /= minibatch_count
        distribution_loss /= minibatch_count
        side_info_loss /= minibatch_count
        if self.distributed:
            loss_collection_work = dist.all_reduce(
                loss, op=dist.ReduceOp.SUM, async_op=True
            )
            reconstruction_loss_collection_work = dist.all_reduce(
                reconstruction_loss, op=dist.ReduceOp.SUM, async_op=True
            )
            distribution_loss_collection_work = dist.all_reduce(
                distribution_loss, op=dist.ReduceOp.SUM, async_op=True
            )
            side_info_loss_collection_work = dist.all_reduce(
                side_info_loss, op=dist.ReduceOp.SUM, async_op=True
            )

        # Update model parameters
        self.scalar.step(self.optimizer)
        self.scalar.update()
        self.learning_rate_scheduler.step()

        if self.distributed:
            loss_collection_work.wait()
            reconstruction_loss_collection_work.wait()
            distribution_loss_collection_work.wait()
            side_info_loss_collection_work.wait()
            loss /= self.world_size
            reconstruction_loss /= self.world_size
            distribution_loss /= self.world_size
            side_info_loss /= self.world_size

        return {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "distribution_loss": distribution_loss.item(),
            "side_info_loss": distribution_loss.item(),
        }

    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        minibatches = split_batch_into_minibatches(
            batch=batch, minibatch_size=self.config["validation_minibatch_size"]
        )
        loss = torch.Tensor([0.0]).to(self.device)
        reconstruction_loss = torch.Tensor([0.0]).to(self.device)
        distribution_loss = torch.Tensor([0.0]).to(self.device)
        side_info_loss = torch.Tensor([0.0]).to(self.device)
        minibatch_count = torch.Tensor([len(minibatches)]).to(self.device)
        with torch.no_grad():
            for i, minibatch in enumerate(minibatches):
                minibatch = self.tokenize_batch(minibatch)
                with autocast(device_type=self.device.type):
                    minibatch_losses = self.calculate_loss(
                        tokens=minibatch,
                    )
                loss += minibatch_losses["loss"].detach()
                reconstruction_loss += minibatch_losses["reconstruction_loss"]
                distribution_loss += minibatch_losses["distribution_loss"]
                side_info_loss += minibatch_losses["side_info_loss"]
            loss /= minibatch_count
            reconstruction_loss /= minibatch_count
            distribution_loss /= minibatch_count
            side_info_loss /= minibatch_count

            if self.distributed:
                loss_collection_work = dist.all_reduce(
                    loss, op=dist.ReduceOp.SUM, async_op=True
                )
                reconstruction_loss_collection_work = dist.all_reduce(
                    reconstruction_loss, op=dist.ReduceOp.SUM, async_op=True
                )
                distribution_loss_collection_work = dist.all_reduce(
                    distribution_loss, op=dist.ReduceOp.SUM, async_op=True
                )
                side_info_loss_collection_work = dist.all_reduce(
                    side_info_loss, op=dist.ReduceOp.SUM, async_op=True
                )
                loss_collection_work.wait()
                reconstruction_loss_collection_work.wait()
                distribution_loss_collection_work.wait()
                side_info_loss_collection_work.wait()
                loss /= self.world_size
                reconstruction_loss /= self.world_size
                distribution_loss /= self.world_size
                side_info_loss /= self.world_size

        return {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "distribution_loss": distribution_loss.item(),
            "side_info_loss": side_info_loss.item(),
        }

    def run_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        step_function: Callable[
            [torch.nn.Module, Dict[str, Any]], Dict[str, torch.Tensor]
        ],
        name: str = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        batch_losses = {
            "loss": [],
            "reconstruction_loss": [],
            "distribution_loss": [],
            "side_info_loss": [],
        }
        if verbose and self.rank == 0:
            data_loader = tqdm(data_loader, desc=name)
        for batch in data_loader:
            losses = step_function(batch)

            # Update metric tracking
            for key in batch_losses.keys():
                batch_losses[key].append(losses[key])
            if verbose and self.rank == 0:
                # TODO factor out magic numbers
                data_loader.set_postfix(
                    {
                        "Loss": "{:4.3f}".format(np.mean(batch_losses["loss"][-300:])),
                        "Recon": "{:4.3f}".format(
                            np.mean(batch_losses["reconstruction_loss"][-300:])
                        ),
                        "Dist": "{:4.3f}".format(
                            np.mean(batch_losses["distribution_loss"][-300:])
                        ),
                        "Side": "{:4.3f}".format(
                            np.mean(batch_losses["side_info_loss"][-300:])
                        ),
                    }
                )
        return {
            key: torch.Tensor(batch_losses[key]).mean().item()
            for key in batch_losses.keys()
        }

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
            training_losses = self.run_epoch(
                data_loader=training_dataloader,
                step_function=self.training_step,
                name="Training",
                verbose=self.verbose,
            )
            self.state["training_history"]["loss"].append(training_losses["loss"])
            self.state["training_history"]["reconstruction_loss"].append(
                training_losses["reconstruction_loss"]
            )
            self.state["training_history"]["distribution_loss"].append(
                training_losses["distribution_loss"]
            )
            self.state["training_history"]["side_info_loss"].append(
                training_losses["side_info_loss"]
            )
            self.model.eval()
            with torch.no_grad():
                validation_losses = self.run_epoch(
                    data_loader=validation_dataloader,
                    step_function=self.validation_step,
                    name="Validation",
                    verbose=self.verbose,
                )
            self.state["validation_history"]["loss"].append(validation_losses["loss"])
            self.state["validation_history"]["reconstruction_loss"].append(
                validation_losses["reconstruction_loss"]
            )
            self.state["validation_history"]["distribution_loss"].append(
                validation_losses["distribution_loss"]
            )
            self.state["validation_history"]["side_info_loss"].append(
                validation_losses["side_info_loss"]
            )
            self.checkpoint(
                best_model=validation_losses["loss"]
                <= min(self.state["validation_history"]["loss"])
            )

        self._log(
            f"Finished training. The best model can be found at {self.checkpoint_directory}."
        )
        return min(self.state["validation_history"]["loss"])
