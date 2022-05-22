from typing import List

import numpy as np

STOPPING_EPOCHS = 4


def velocity_early_stopping(losses: List[float]) -> bool:
    if len(losses) < STOPPING_EPOCHS:
        return False
    losses = np.array(losses)
    deltas = losses[1:] - losses[:-1]
    velocity_avg = np.mean(deltas)

    return all(losses[-STOPPING_EPOCHS:] < velocity_avg)


def absolute_early_stopping(losses: List[float]) -> bool:
    if len(losses) < STOPPING_EPOCHS:
        return False
    losses = np.array(losses)
    deltas = losses[1:] - losses[:-1]

    return all(deltas[-STOPPING_EPOCHS:] > 0)
