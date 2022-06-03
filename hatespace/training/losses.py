from random import randrange
from typing import Callable

import torch
import geomloss


class SampledDirichletLoss:
    """Dirichlet loss for sampled distributions.

    Calculates a distance metric between input points and sampled dirichlet distribution.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        num_samples: int = 100,
        sample_distance_function: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = geomloss.SamplesLoss(),
    ) -> None:
        """Initialize the SampledDirichletLoss.

        Creates a new instance of the loss, setting the alpha value for the desired
        target distribution, the number of samples to draw and optionally the distance
        function to use for calculating the loss. Larger values of `num_samples` will
        increase the accuracy of the loss, but take more time and memory to compute.

        Args:
            alpha (:obj:`float`): Distribution density for the target dirichlet.
            num_samples (:obj:`int`): Number of samples to draw from the target dirichlet.
            sample_distance_function (:type:`Callable`, optional): Function to evaluate the distribution distances.
        """
        self.alpha = alpha
        self.num_samples = num_samples
        self.sample_distance_function = sample_distance_function
        self.device = "cpu"

        # Set dynamically during __call__
        self.num_dimensions = None
        self.distribution_sampler = None

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """Calculates dirichlet distribution distance.

        Calculates the distance from the given points to a sampled dirichlet
        distribution with density of `self.alpha`. Points are expected to be a float
        type tensor of shape (num_points, num_dimensions). `num_points` number of
        sampled points will then be drawn from the dirichlet distribution to calculate
        the distance.

        Args:
            points (:obj:`torch.Tensor`): A set of points.

        Returns:
            torch.Tensor: Approximate distance of points from a dirichlet distribution.
        """
        assert (
            len(points.shape) == 2
        ), "Incorrect number of dimensions for SampledDirichletLoss!\nInputs must be 2 dimensional tensors with the format (batch_size, num_dimensions)."
        batch_size, num_dimensions = points.shape
        if not num_dimensions == self.num_dimensions:
            self.distribution_sampler = self._get_sampler(num_dimensions)
        # TODO Figure out how many samples is appropriate
        dirichlet_sample = self.distribution_sampler.sample([self.num_samples]).to(
            self.device
        )
        return self.sample_distance_function(points, dirichlet_sample)

    def to(self, device: str) -> "SampledDirichletLoss":
        """Moves the loss function to the specified computation device

        Args:
            device (:obj:`str`): Device to move to.
        """
        self.device = device
        return self

    def _get_sampler(self, num_dimensions: int) -> torch.distributions.Dirichlet:
        return torch.distributions.dirichlet.Dirichlet(
            torch.tensor([self.alpha for i in range(num_dimensions)])
        )


class SequenceLoss:
    """Sequence loss for seq2seq models.

    Calculates the cross entropy loss between a sequence of logits and a set of target
    classes.
    """

    def __init__(
        self,
    ) -> None:
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the sequence loss.

        Calculates the cross entropy loss between a sequence of logits and a set of
        target classes. The entropy is averaged over the total number of logits within
        the given batch.

        Logits are expected as a float type tensor in the format
        `(batch_size, sequence_length, vocab_size)`,
        and targets are expected as a long type in the format
        `(batch_size, sequence_length)`,
        where each entry is an integer between `0` and `vocab_size`.

        Args:
            logits (:obj:`torch.Tensor`): A sequence of model predictions.
            targets (:obj:`torch.Tensor`): A sequence of target classes.

        Returns:
            torch.Tensor: The resulting cross entropy loss.
        """
        targets = targets.view(-1)
        total_token_size = targets.shape[0]
        return self.loss_fn(logits.view(total_token_size, -1), targets)
