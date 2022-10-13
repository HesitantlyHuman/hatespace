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
        predicted_sequence_logits, embeddings = (
            model_outputs.logits,
            model_outputs.embeddings,
        )
        del model_outputs

        return predicted_sequence_logits, embeddings

    def calculate_loss(
        self,
        tokens: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        model_predictions, embeddings = self.batch_prediction(tokens=tokens)
        loss = self.loss_function(
            model_predictions,
            tokens["input_ids"],
            embeddings,
        )
        return loss
