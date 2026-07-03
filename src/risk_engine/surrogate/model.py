"""PyTorch surrogate model for Expected Credit Loss approximation."""
import torch
from torch import nn

class ECLSurrogate(nn.Module):
    """
    Multi-layer perceptron mapping macro features to portfolio ECL.

    Architecture: 3 -> 64 -> 32 -> 1 with ReLU activations.
    """

    INPUT_DIM = 3
    HIDDEN_DIMS = (64, 32)
    OUTPUT_DIM = 1

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.INPUT_DIM, self.HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_DIMS[0], self.HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_DIMS[1], self.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
