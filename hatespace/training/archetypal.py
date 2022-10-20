from typing import Dict, Tuple

import torch
from hatespace.training.trainer import HatespaceTrainer


class ArchetypalTrainer(HatespaceTrainer):
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
