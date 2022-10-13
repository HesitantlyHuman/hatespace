from typing import Callable, Union, Optional

import torch
import geomloss


class HatespaceMultiCriterion:
    def __init__(
        self,
        reconstruction_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        distribution_loss: Callable[[torch.Tensor], torch.Tensor],
        side_info_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        reconstruction_loss_weight: float = 1.0,
        distribution_loss_weight: float = 1.0,
        side_info_loss_weight: float = 1.0,

    ):
        self.reconstruction_loss = reconstruction_loss
        self.distribution_loss = distribution_loss
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.distribution_loss_weight = distribution_loss_weight

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction_loss = self.reconstruction_loss(logits, targets)
        distribution_loss = self.distribution_loss(embeddings)

        return (
            self.reconstruction_loss_weight * reconstruction_loss
            + self.distribution_loss_weight * distribution_loss
        )


# TODO consider creating custom sinkhorn loss and testing speed
# see https://dfdazac.github.io/sinkhorn.html
# and also https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/Pytorch_Wasserstein.ipynb
# the second uses a custom cuda kernel, which may give significant speedups


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
        _, num_dimensions = points.shape
        if not num_dimensions == self.num_dimensions:
            self.distribution_sampler = self._get_sampler(num_dimensions)
        dirichlet_sample = self.distribution_sampler.sample([self.num_samples]).to(
            self.device
        )
        return self.sample_distance_function(points, dirichlet_sample)

    def to(self, device: Union[str, torch.device]) -> "SampledDirichletLoss":
        """Moves the loss function to the specified computation device

        Args:
            device (:obj:`str`): Device to move to.
        """
        self.device = device
        return self

    def cuda(
        self, device: Optional[Union[int, torch.device]] = None
    ) -> "SampledDirichletLoss":
        """Moves the loss function to the specified GPU.

        Args:
            gpu (:obj:`int`): GPU to move to.
        """
        if device is None:
            device = torch.cuda.current_device()
        if isinstance(device, int):
            device = torch.device("cuda", device)
        self.device = device
        return self

    def _get_sampler(self, num_dimensions: int) -> torch.distributions.Dirichlet:
        return torch.distributions.dirichlet.Dirichlet(
            torch.tensor([self.alpha for i in range(num_dimensions)])
        )

    def __repr__(self) -> str:
        return f"SampledDirichletLoss(alpha={self.alpha}, num_samples={self.num_samples}, sample_distance_function={self.sample_distance_function})"


class SequenceLoss:
    """Sequence loss for seq2seq models.

    Calculates the cross entropy loss between a sequence of logits and a set of target
    classes.
    """

    def __init__(
        self,
        ignore_index: int = None,
        reduction: str = "mean",
    ) -> None:
        if ignore_index is not None:
            self.loss_fn = torch.nn.NLLLoss(
                ignore_index=ignore_index, reduction=reduction
            )
        else:
            self.loss_fn = torch.nn.NLLLoss(reduction=reduction)

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
        predictions = torch.nn.functional.log_softmax(logits, dim=2)
        predictions = predictions[:, :-1, :].contiguous()
        targets = targets[:, 1:]

        rearranged_output = predictions.view(
            predictions.shape[0] * predictions.shape[1], -1
        )
        rearranged_target = targets.contiguous().view(-1)

        loss = self.loss_fn(rearranged_output, rearranged_target)

        return loss


class SideInfoLoss:
    """Side information loss for linear head.

    Calculates the binary cross entropy loss between logits and target
    """

    def __init__(
        self,
        pos_weight: torch.Tensor = None,
        reduction: str = "mean",
    ) -> None:
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(logits, target)
        return loss
