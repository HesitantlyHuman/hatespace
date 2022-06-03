from transformers.trainer import Trainer


class HatespaceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs)
