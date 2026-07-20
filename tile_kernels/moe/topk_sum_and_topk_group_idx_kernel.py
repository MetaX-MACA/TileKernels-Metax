import os

import tilelang
import torch
from tilelang import language as T

from tile_kernels.moe.common import get_topk_group_idx
from tile_kernels.utils import align


@T.macro
def get_topk_group_idx_wave64(
    scores_shared: T.SharedBuffer,
    topk_group_idx_shared: T.SharedBuffer,
    group_scores_shared: T.SharedBuffer,
    num_groups: int,
    num_experts_per_group: int,
    num_topk_groups: int,
    num_topk_sum: int,
    num_vectorize_for_grouped_expert: int,
):
    thread_idx = T.get_thread_binding()
    token_idx = thread_idx // 32
    lane_idx = thread_idx % 32
    scores_vec_local = T.alloc_local((num_vectorize_for_grouped_expert,), dtype=T.float32)

    top1_var = T.alloc_var(dtype=T.float32, init=-T.infinity(T.float32))
    top2_var = T.alloc_var(dtype=T.float32, init=-T.infinity(T.float32))
    topk_sum_var = T.alloc_var(dtype=T.float32, init=-T.infinity(T.float32))
    count_var = T.alloc_var(dtype=T.int32, init=0)

    if lane_idx < num_groups:
        num_vec_experts_per_group = num_experts_per_group // num_vectorize_for_grouped_expert
        for i in T.unroll(num_vec_experts_per_group):
            for j in T.vectorized(num_vectorize_for_grouped_expert):
                vec_idx = (i + lane_idx) % num_vec_experts_per_group
                scores_vec_local[j] = scores_shared[
                    token_idx, lane_idx * num_experts_per_group + vec_idx * num_vectorize_for_grouped_expert + j
                ]
                if scores_vec_local[j] > top1_var:
                    top2_var = top1_var
                    top1_var = scores_vec_local[j]
                elif scores_vec_local[j] > top2_var:
                    top2_var = scores_vec_local[j]
        topk_sum_var = T.Select(num_topk_sum == 1, top1_var, top1_var + top2_var)

    group_scores_shared[token_idx, lane_idx] = topk_sum_var
    T.sync_threads()

    if lane_idx < num_groups:
        for i in T.unroll(num_groups):
            other_topk_sum = group_scores_shared[token_idx, i]
            if other_topk_sum > topk_sum_var or (other_topk_sum == topk_sum_var and i < lane_idx):
                count_var += 1
        if count_var < num_topk_groups:
            topk_group_idx_shared[token_idx, count_var] = lane_idx

    T.sync_threads()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_topk_sum_and_topk_group_idx_kernel(
    num_groups: int,
    num_experts_per_group: int,
    num_topk_groups: int,
    num_topk_sum: int,
):
    num_experts = num_experts_per_group * num_groups
    use_wave64 = num_experts % 64 == 0
    num_threads = 64 if use_wave64 else 32
    num_aligned_experts = align(num_experts, num_threads)
    num_tokens_per_block = num_threads // 32

    assert num_groups <= 32, f'num_groups ({num_groups}) must be <= warp size (32)'

    # Make sure that the number of experts per group is divisible by vectorization size.
    num_vectorize_for_grouped_expert = 4
    while num_experts_per_group % num_vectorize_for_grouped_expert != 0:
        num_vectorize_for_grouped_expert //= 2
    assert num_experts_per_group % num_vectorize_for_grouped_expert == 0

    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def topk_sum_and_topk_group_idx_kernel(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        group_topk_idx: T.Tensor[(num_tokens, num_topk_groups), T.int64],
    ):
        with T.Kernel(T.ceildiv(num_tokens, num_tokens_per_block), threads=num_threads) as pid:
            scores_shared = T.alloc_shared((num_tokens_per_block, num_aligned_experts), T.float32)
            topk_group_idx_shared = T.alloc_shared((num_tokens_per_block, num_topk_groups), T.int32)
            group_scores_shared = T.alloc_shared((num_tokens_per_block, 32), T.float32)

            thread_idx = T.get_thread_binding()
            token_idx = thread_idx // 32
            lane_idx = thread_idx % 32
            token_base = pid * num_tokens_per_block
            token = token_base + token_idx

            if use_wave64:
                if token_base + 1 < num_tokens:
                    T.copy(scores[token_base, 0], scores_shared)
                elif token_base < num_tokens:
                    for i in T.Parallel(num_aligned_experts):
                        scores_shared[0, i] = T.Select(i < num_experts, scores[token_base, i], -T.infinity(T.float32))
                        scores_shared[1, i] = -T.infinity(T.float32)
                else:
                    for i in T.Parallel(num_aligned_experts):
                        scores_shared[token_idx, i] = -T.infinity(T.float32)
                T.sync_threads()

                get_topk_group_idx_wave64(
                    scores_shared=scores_shared,
                    topk_group_idx_shared=topk_group_idx_shared,
                    group_scores_shared=group_scores_shared,
                    num_groups=num_groups,
                    num_experts_per_group=num_experts_per_group,
                    num_topk_groups=num_topk_groups,
                    num_topk_sum=num_topk_sum,
                    num_vectorize_for_grouped_expert=num_vectorize_for_grouped_expert,
                )
            else:
                T.copy(scores[pid, 0], scores_shared)
                T.sync_warp()
                get_topk_group_idx(
                    scores_shared=scores_shared,
                    topk_group_idx_shared=topk_group_idx_shared,
                    num_groups=num_groups,
                    num_experts_per_group=num_experts_per_group,
                    num_topk_groups=num_topk_groups,
                    num_topk_sum=num_topk_sum,
                    num_vectorize_for_grouped_expert=num_vectorize_for_grouped_expert,
                )

            if token < num_tokens and lane_idx < num_topk_groups:
                group_topk_idx[token, lane_idx] = topk_group_idx_shared[token_idx, lane_idx]

    return topk_sum_and_topk_group_idx_kernel


def topk_sum_and_topk_group_idx(scores: torch.Tensor, num_topk_sum: int, num_topk_groups: int) -> torch.Tensor:
    """Return top-``num_topk_groups`` group indices ranked by intra-group top-k sum.

    For each token, this function computes a group score by summing the largest
    ``num_topk_sum`` expert scores within each group, then returns the indices of
    the groups with the highest summed values.

    Args:
        scores: Contiguous float32 tensor with shape
            ``(num_tokens, num_groups, num_experts_per_group)``.
        num_topk_sum: Number of top expert scores to sum per group. Only ``1``
            and ``2`` are supported.
        num_topk_groups: Number of highest-scoring groups to select per token.

    Returns:
        ``torch.int64`` tensor of shape ``(num_tokens, num_topk_groups)``
        containing selected group indices for each token.
    """
    assert scores.dim() == 3 and scores.is_contiguous() and scores.dtype == torch.float32
    num_tokens, num_groups, num_experts_per_group = scores.shape
    assert num_topk_sum <= num_experts_per_group and num_topk_sum in (1, 2) and num_topk_groups <= num_groups

    kernel = get_topk_sum_and_topk_group_idx_kernel(num_groups, num_experts_per_group, num_topk_groups, num_topk_sum)
    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    topk_group_idx = torch.empty(num_tokens, num_topk_groups, dtype=torch.int64, device=scores.device)
    if num_tokens == 0:
        return topk_group_idx

    kernel(scores.view(num_tokens, -1), topk_group_idx)
    return topk_group_idx
