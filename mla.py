import torch
from torch import nn

from layers.column_parallel_linear import ColumnParallelLinear
from layers.linear import Linear
from model_args import ModelArgs

WORLD_SIZE = 1


class MLA(nn.Module):
    """
    Multi-head Latent Attention

    Attributes:
        dim (int): Dimensionality of each token's embedding
        n_heads (int): Number of attention heads
        n_local_heads (int): Number of local attention heads for distributed systems
        q_lora_rank (int): Rank for low-rank query projection
        kv_lora_rank (int): Rank for low-rank key-value projection
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections
        qk_rope_head_dim (int): Dimensionality of positional query/key projections
        qk_head_dim (int): Total dimensionality of query/key projections
        v_head_dim (int): Dimensionality of value projections
        softmax_scale (float): Scaling factor for softmax in attention computation
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // WORLD_SIZE
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # Note that moving down, my naming convention follows Welch Labs' video instead of the paper as I find it
        # more intuitive.

        # If q_lora_rank == 0, that means that we are not projecting X into the latent space before getting queries.
        # If it's non-zero, then we are also projecting X into the latent space (note that this is a different latent
        # space than that of KV) before converting it to queries.
        if self.q_lora_rank == 0:
            # ColumnParallelLinear is same as linear on one GPU. If there are multiple GPUs, then it splits the columns
            # across multiple GPUs (TensorParallel)
            self.w_q = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.w_dq = Linear(self.dim, self.q_lora_rank)  # Down projects the queries into latent space
            self.normalize_q = RMSNorm(self.q_lora_rank)  # Normalize it along the q_lora_rank (each query) dimension
            self.w_uq = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # If you look at the paper's equations, W_dkv and W_kr are both multiplied by `h_t`, the input (Eqn 1 and 3).
        # And in DeepSeek's implementation, they combine the two following matrices into one. But I'm writing it for
        # clarity, so I'll keep it separate.
        self.w_dkv = Linear(self.dim, self.kv_lora_rank)
        self.w_kr = Linear(self.dim, self.qk_rope_head_dim)
        self.normalize_kv = RMSNorm(self.kv_lora_rank)

        # Again, if you look at equations 2 and 5, the weights are being multiplied by the same input. And the official
        # implementation combines these as well, but I'll keep it separate.
        self.w_uk = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * self.qk_nope_head_dim)
        self.w_uv = ColumnParallelLinear(self.kv_lora_rank, self.v_head_dim)
        self.w_o = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5