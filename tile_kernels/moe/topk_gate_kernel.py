import os

import tilelang
import torch
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_topk_gate_kernel(num_experts: int, num_topk: int):
    num_tokens = T.dynamic('num_tokens')
    subgroup_size = 32
    num_threads = 64
    num_tokens_per_block = num_threads // subgroup_size
    num_experts_per_lane = (num_experts + subgroup_size - 1) // subgroup_size

    @T.prim_func
    def topk_gate_kernel(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        with T.Kernel(T.ceildiv(num_tokens, num_tokens_per_block), threads=num_threads) as pid:
            thread_idx = T.get_thread_binding()
            token_idx = thread_idx // subgroup_size
            lane_idx = thread_idx % subgroup_size
            token = pid * num_tokens_per_block + token_idx

            key_local = T.alloc_local((num_experts_per_lane,), T.uint64)
            idx_local = T.alloc_local((num_experts_per_lane,), T.int32)
            topk_key_var = T.alloc_var(T.uint64)
            other_key = T.alloc_var(T.uint64)

            for i in T.unroll(num_experts_per_lane):
                expert_idx = lane_idx + i * subgroup_size
                idx_local[i] = expert_idx
                if token < num_tokens and expert_idx < num_experts:
                    score_bits = T.reinterpret(scores[token, expert_idx], T.uint32)
                    magnitude = score_bits & T.uint32(0x7FFFFFFF)
                    is_nan = magnitude > T.uint32(0x7F800000)
                    is_negative = ((score_bits & T.uint32(0x80000000)) != 0) & (magnitude != 0)
                    normalized_bits = T.Select(magnitude == 0, T.uint32(0), score_bits)
                    ordered_score = T.Select(
                        is_nan,
                        T.Select(is_negative, T.uint32(0), T.uint32(0xFFFFFFFF)),
                        T.Select(is_negative, ~normalized_bits, normalized_bits ^ T.uint32(0x80000000)),
                    )
                    key_local[i] = (T.uint64(ordered_score) << 32) | T.uint64(T.uint32(~expert_idx))
                else:
                    key_local[i] = T.uint64(0)

            # Repeated argmax with one 32-lane tournament per selected expert.
            for k in T.unroll(num_topk):
                topk_key_var = T.uint64(0)
                for i in T.unroll(num_experts_per_lane):
                    topk_key_var = T.max(topk_key_var, key_local[i])
                for i in T.unroll(5):
                    other_key = T.shfl_xor(topk_key_var, 1 << i, width=subgroup_size)
                    topk_key_var = T.max(topk_key_var, other_key)
                topk_idx_local = T.cast(T.uint32(~T.uint32(topk_key_var)), T.int32)
                if token < num_tokens and lane_idx == 0:
                    topk_idx[token, k] = topk_idx_local
                for i in T.unroll(num_experts_per_lane):
                    if idx_local[i] == topk_idx_local:
                        key_local[i] = T.uint64(0)

    return topk_gate_kernel


def topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """Select the top-k experts per token from scores.

    Args:
        scores (torch.Tensor): Gating logits or scores with shape
            ``[num_tokens, num_experts]``. Higher values indicate stronger
            routing preference.
        num_topk (int): Number of experts to select per token. Must satisfy
            ``1 <= num_topk <= num_experts``.

    Returns:
        torch.Tensor: Top-k expert indices with shape ``[num_tokens, num_topk]``
            and ``torch.int64``. Each row contains the selected
            expert indices for the corresponding token.

    Notes:
        - Always return the smaller index when there are ties.
        - The output is always contiguous.
    """
    assert scores.dim() == 2 and scores.is_contiguous() and scores.dtype == torch.float32
    num_tokens, num_experts = scores.shape
    assert num_topk <= num_experts, f'num_topk ({num_topk}) must be <= num_experts ({num_experts})'
    topk_idx = torch.empty((num_tokens, num_topk), dtype=torch.int64, device=scores.device)
    if num_tokens == 0:
        return topk_idx

    kernel = get_topk_gate_kernel(num_experts, num_topk)

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    kernel(scores, topk_idx)
    return topk_idx
