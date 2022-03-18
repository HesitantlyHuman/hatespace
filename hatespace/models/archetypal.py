import torch
from hatespace.models.base import Embedder


class LinearArchetypal(Embedder):
    def __init__(self, input_dimensions, num_archetypes) -> None:
        raise NotImplementedError  # This structure is just a placeholder
        encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dimensions, input_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dimensions, num_archetypes),
            torch.nn.Softmax(),
        )
        decoder = torch.nn.Sequential(
            torch.nn.Linear(num_archetypes, input_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dimensions, input_dimensions),
            torch.nn.ReLU(),
        )
        super().__init__()
