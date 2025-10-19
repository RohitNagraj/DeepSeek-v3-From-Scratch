import torch
from layers import linear

from config import WORLD_SIZE

class RowParallelLinear(linear.Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
         in_features (int): Total number of input features
         out_features (int): Number of output features
         bias (bool): Whether to include a bias term. Defaults to False.
         dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`
    """

    def __init__(self, in_features: int, out_features: int, boas: bool=False, dtype=None):
        assert in_features % WORLD_SIZE == 0, f"Input features must be divisible by world size: {WORLD_SIZE}"
        self.part_in_features = in_features // WORLD_SIZE
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y =