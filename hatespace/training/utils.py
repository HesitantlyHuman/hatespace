from typing import Iterable, List, Union, Generator, Any
import torch
import numpy as np

STOPPING_EPOCHS = 50


def absolute_early_stopping(losses: List[Union[float, torch.Tensor]]) -> bool:
    if len(losses) < STOPPING_EPOCHS:
        return False
    losses = np.array(losses)
    deltas = losses[1:] - losses[:-1]

    return all(deltas[-STOPPING_EPOCHS:] > 0)


class GeneratorSlice:
    def __init__(self, generator: Generator, n: int) -> None:
        self.generator = generator
        self.n = n
        self._current_n = 0

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterable:
        return self

    def __next__(self) -> Any:
        if self._current_n < self.n:
            self._current_n += 1
            return next(self.generator)
        else:
            raise StopIteration
