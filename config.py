from typing import Literal

WORLD_SIZE = 1
BLOCK_SIZE = 128
GEMM_IMPL: Literal["bf16", "fp8"] = "bf16"
ATTN_IMPL: Literal["naive", "absorb"] = "absorb"
