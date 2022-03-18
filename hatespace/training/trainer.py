from transformers import Trainer
from torch.utils.data import DataLoader
from hatespace.datasets.base import DataLoader as IronmarchDataLoader

# TODO: We may want to modify this class so that we can easily switch
# between using sinkhorn, kl, vertex loss and others.
class IronmarchTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        raise NotImplementedError

    def get_train_dataloader(self) -> DataLoader:
        return IronmarchDataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self) -> DataLoader:
        return IronmarchDataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
