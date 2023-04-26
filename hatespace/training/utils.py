from typing import Iterable, List, Union, Generator, Any, Dict
import torch
import numpy as np
import uuid
import hatespace

STOPPING_EPOCHS = 50


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    hatespace.datasets.base.functional._seed = seed


def generate_experiment_name() -> str:
    try:
        from unique_names_generator import get_random_name
        from unique_names_generator.data import ADJECTIVES, ANIMALS

        return get_random_name(
            combo=[ADJECTIVES, ANIMALS], separator="-", style="lowercase"
        )
    except ImportError:
        return uuid.uuid4().hex[:6]


def split_batch_into_minibatches(
    batch: Dict[str, Any], minibatch_size: int = 2
) -> List[Dict[str, Any]]:
    minibatches = []
    for i in range(0, len(batch["data"]), minibatch_size):
        minibatch = {key: value[i : i + minibatch_size] for key, value in batch.items()}
        minibatches.append(minibatch)
    return minibatches


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


def _convert_bytes_to_gb(bytes: int) -> float:
    return bytes / (1024**3)


def report_cuda_memory_info() -> str:
    """
    Returns a string with information about the CUDA memory usage.
    Reported memory usage is in GB.
    """

    return (
        "-- Allocated Memory --\n"
        + f"Current: {_convert_bytes_to_gb(torch.cuda.memory_allocated()):0.2f} GB\n"
        + f"Max: {_convert_bytes_to_gb(torch.cuda.max_memory_allocated()):0.2f} GB\n"
        + "-- Cached Memory --\n"
        + f"Current: {_convert_bytes_to_gb(torch.cuda.memory_cached()):0.2f} GB\n"
        + f"Max: {_convert_bytes_to_gb(torch.cuda.max_memory_cached()):0.2f} GB\n"
        + "-- Reserved Memory --\n"
        + f"Current: {_convert_bytes_to_gb(torch.cuda.memory_reserved()):0.2f} GB\n"
        + f"Max: {_convert_bytes_to_gb(torch.cuda.max_memory_reserved()):0.2f} GB\n"
    )


if __name__ == "__main__":
    for i in range(100):
        print(generate_experiment_name())
