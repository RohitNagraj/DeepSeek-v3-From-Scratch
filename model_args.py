from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters

    Attributes:
        max_batch_size (int): Maximum batch size
        max_seq_len (int): Maximum sequence length
        dtype (Literal["bf16", "fp8"]): Data type for computations
        vocab_size (int): Vocabulary size
        dim (int): Model dimension
        inter_dim (int): Intermediate dimension for MLP layers
        moe_inter_dim (int): Intermediate dimension for MoE layers
        n_layers (int): Number of transformer layers
        n_dense_layers (int): Number of dense layers in the model
        n_heads (int): Number of attention heads
        n_routed_experts (int): Number of routed experts for MoE layers
        n_shared_experts (int): Number of shared experts for MoE layers
        n_activated_experts (int): Number of activated experts in MoE layers
        n_expert_groups (int): Number of expert groups
        n_limited_groups (int): Number of limited groups for MoE routing
        score_func (Literal["softmax, "sigmoid"]): Scoring fucntion for MoE routing
        route_scale (float): Scaling factor for routing scores
        q_lora_rank (int): LoRA rank for query projections # Note this this has nothing to do with QLoRA
        kv_lora_rank (int): LoRA rank for key-value projections
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings
        v_head_dim (int): Dimension for value projections
        original_seq_len (int): Original sequence length
        rope_thta (float): Base for rotary prositional encoding
        rope_factor (float): Scaling factor for extended sequence lengths
        beta_fast (int): Fast beta correction factor
        beta_slow (int): Slow beta correction factor
        mscale (float): Scaling factor for extended attention
    """

    # Note that these values are for the 16B version. These will change for the original 671B.

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16

    #MoE
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0

    # MLA
    q_lora_rank: int = 0 # This means we aren't really using the Latent space for Q. This is exactly same as MHA for Q.
    kv_lora_rank: int = 512 # Pulling down the `dim = 2048` to 512 space for KV.
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # YaRN (Efficient Context Window Extension)
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


