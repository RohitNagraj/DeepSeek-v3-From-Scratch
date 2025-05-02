import torch
from torch import nn
import torch.nn.functional as F

from config import BLOCK_SIZE, GEMM_IMPL


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        use_bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, use_bias: bool = False, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))

        # element_size() returns the number of bytes for each element.
        if self.weight.element_size() == 1:
            # This condition is true when dtype=`torch.float8`
            # We also add scale which is the scaling factor when quantized
            scale_out_features = (out_features + BLOCK_SIZE - 1) // BLOCK_SIZE
            scale_in_features = (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            # Case when dtype=`bfloat16`
            self.register_parameter("scale", None)

        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
             x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return self.linear(x)

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a linear transformation to the incoming data: y = xA^T + b.
        This function supports specialized implementations based on quantization and tensor formats.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: The result of the linear transformation, which may involve quantization-aware computations
            depending on the input parameters.

        Notes:
            - If `self.weight` is not quantized (16 bit or higher), we directly multiply.
            - Else
                - If `GEMM_IMPL == "bf16"`, we dequantize `self.weight` and compute `xA^T` in BF16.
                - Else, we quantize the activations `x` to FP8 and carry out GEMM in FP8 (Hardware support on Hopper and above).
        """
        if self.weight.element_size() > 1:
            return F.linear(x, self.weight, self.bias)

        elif GEMM_IMPL == "bf16":
            weight = weight_dequant(self.weight, self.weight.scale)
            return F.linear(x, weight, self.bias)
        else:
            x, scale = act_quant(x, BLOCK_SIZE)
            y = fp8_gemm(x, scale, self.weight, self.weight.scale)
            if self.bias is not None:
                y += self.bias
            return y