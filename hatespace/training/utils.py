from typing import List, Union
import torch
import numpy as np

STOPPING_EPOCHS = 50


def absolute_early_stopping(losses: List[Union[float, torch.Tensor]]) -> bool:
    if len(losses) < STOPPING_EPOCHS:
        return False
    losses = np.array(losses)
    deltas = losses[1:] - losses[:-1]

    return all(deltas[-STOPPING_EPOCHS:] > 0)
