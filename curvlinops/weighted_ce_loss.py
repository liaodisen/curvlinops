"""Cross-entropy loss with data-dependent weights."""

from torch import Tensor, ones
from torch.nn import CrossEntropyLoss, Parameter


class CrossEntropyLossWeighted(CrossEntropyLoss):

    def __init__(self, num_data: int, *args, **kwargs) -> None:
        """Initialize CrossEntropyLossWeighted.

        Args:
            num_data: Number of data points for which weights are defined.
            *args: Additional arguments passed to CrossEntropyLoss.
            **kwargs: Additional keyword arguments passed to CrossEntropyLoss.

        Raises:
            ValueError: If num_data is not a positive integer.
        """
        if not isinstance(num_data, int) or num_data <= 0:
            raise ValueError(f"num_data must be a positive integer, got {num_data}")
        
        super().__init__(*args, **kwargs)
        self._num_data = num_data
        self.data_weights = Parameter(0.5 * ones(num_data))

    def forward(self, input: Tensor, target_and_data_idx: Tensor) -> Tensor:
        """Forward pass with data-dependent weights.

        Args:
            target_and_data_idx: Has shape (N, 2) where N is the batch size.
                First entry is the label used for cross-entropy, second entry
                is the data point's index, which we will use to look up the
                weight.
        """
        # split targets and data point indices
        target, data_idx = target_and_data_idx.split([1, 1], dim=1)
        target = target.squeeze(1)
        data_idx = data_idx.squeeze(1)

        # compute the unreduced loss
        actual_reduction = self.reduction
        self.reduction = "none"
        loss = super().forward(input, target)
        self.reduction = actual_reduction

        # apply the weights
        sigma = self.data_weights[data_idx].clamp(min=0.0, max=1.0)
        loss = loss * sigma

        # do the reduction
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError