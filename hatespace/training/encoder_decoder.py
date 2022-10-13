from typing import Dict

import torch
from hatespace.training.trainer import HatespaceTrainer


class EncoderDecoderTrainer(HatespaceTrainer):
    def batch_prediction(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
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
        tokens: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        input_ids = tokens["input_ids"]
        model_predictions = self.batch_prediction(tokens=tokens)
        loss = self.loss_function(model_predictions, input_ids)

        return loss
