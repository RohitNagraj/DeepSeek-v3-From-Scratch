import torch
from layers import linear

from config import WORLD_SIZE

class ColumnParallelLinear(linear.Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input_features
        out_features (int): Number of output features
        bias (bool): Whether to include a bias term. Defaults to False
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`
    """

    def __init__(self, in_features: int, out_features: int, bias: bool=False, dtype=None):
        assert out_features % WORLD_SIZE == 0, f"Output features must be divisble by world size (word_size={WORLD_SIZE})"
        self.part_out_features = out_features // WORLD_SIZE
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor)  -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation
        """
        y = self.linear(x)
        # If we have multiple GPUs, then y would just be a part of the output.
        return y

