# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import math

import tilelang
import torch
from tilelang import language as T


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


def _make_ptr_tables_batched(
    tensor_lists: list[list[torch.Tensor]],
    device: torch.device,
) -> list[torch.Tensor]:
    sizes = [len(tl) for tl in tensor_lists]
    total = sum(sizes)
    if total == 0:
        return [torch.empty(0, device=device, dtype=torch.int64) for _ in tensor_lists]

    pinned = torch.empty(total, dtype=torch.int64, device='cpu', pin_memory=True)
    offset = 0
    for tl in tensor_lists:
        for i, t in enumerate(tl):
            pinned[offset + i] = t.data_ptr()
        offset += len(tl)

    gpu_buf = pinned.to(device=device, non_blocking=True)

    tables: list[torch.Tensor] = []
    offset = 0
    for sz in sizes:
        tables.append(gpu_buf[offset : offset + sz])
        offset += sz
    return tables


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_multilayer_recompute_kernel(
    mhc_mult: int,
    hidden: int,
    num_layers: int,
    num_post: int,
    n_thr: int = 64,
    h_blk: int = 2048,
) -> tilelang.JITKernel:
    n = T.dynamic('num_tokens')
    h = hidden
    mhc = mhc_mult
    L = num_layers
    L_post = num_post

    h_blk = math.gcd(h_blk, hidden)

    @T.prim_func
    def kernel(
        initial_residual: T.Tensor[(n, mhc, h), T.bfloat16],
        pre_mix_ptrs: T.Tensor[(L,), T.ptr],
        layer_output_ptrs: T.Tensor[(L_post,), T.ptr],
        post_mix_ptrs: T.Tensor[(L_post,), T.ptr],
        comb_mix_ptrs: T.Tensor[(L_post,), T.ptr],
        layer_input_ptrs: T.Tensor[(L,), T.ptr],
        residual_ptrs: T.Tensor[(L_post,), T.ptr],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as i_n:
            res_local = T.alloc_fragment((mhc, h_blk), T.float32)
            new_res_local = T.alloc_fragment((mhc, h_blk), T.float32)
            layer_input_local = T.alloc_fragment(h_blk, T.float32)
            layer_output_local = T.alloc_fragment(h_blk, T.float32)
            pre_mix_local = T.alloc_fragment(mhc, T.float32)
            post_mix_local = T.alloc_fragment(mhc, T.float32)
            comb_mix_local = T.alloc_fragment((mhc, mhc), T.float32)

            for i0_h in T.serial(h // h_blk):
                T.copy(initial_residual[i_n, 0, i0_h * h_blk], res_local)

                for i_layer in T.serial(L_post):
                    layer_output_tensor = T.make_tensor(layer_output_ptrs[i_layer], (n, h), T.bfloat16)
                    pre_mix_tensor = T.make_tensor(pre_mix_ptrs[i_layer], (n, mhc), T.float32)
                    post_mix_tensor = T.make_tensor(post_mix_ptrs[i_layer], (n, mhc), T.float32)
                    comb_mix_tensor = T.make_tensor(comb_mix_ptrs[i_layer], (n, mhc, mhc), T.float32)
                    layer_input_tensor = T.make_tensor(layer_input_ptrs[i_layer], (n, h), T.bfloat16)
                    output_residual_tensor = T.make_tensor(residual_ptrs[i_layer], (n, mhc, h), T.bfloat16)

                    T.copy(pre_mix_tensor[i_n, 0], pre_mix_local)

                    T.clear(layer_input_local)
                    for i_mhc in T.serial(mhc):
                        for i1_h in T.Parallel(h_blk):
                            layer_input_local[i1_h] += pre_mix_local[i_mhc] * res_local[i_mhc, i1_h]

                    T.copy(layer_input_local, layer_input_tensor[i_n, i0_h * h_blk])

                    T.copy(post_mix_tensor[i_n, 0], post_mix_local)
                    T.copy(comb_mix_tensor[i_n, 0, 0], comb_mix_local)
                    T.copy(layer_output_tensor[i_n, i0_h * h_blk], layer_output_local)
                    for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                        new_res_local[i_mhco, i1_h] = post_mix_local[i_mhco] * layer_output_local[i1_h]
                        for i_mhci in T.serial(mhc):
                            new_res_local[i_mhco, i1_h] += comb_mix_local[i_mhci, i_mhco] * res_local[i_mhci, i1_h]

                    T.copy(new_res_local, output_residual_tensor[i_n, 0, i0_h * h_blk])

                    for i_mhc, i1_h in T.Parallel(mhc, h_blk):
                        res_local[i_mhc, i1_h] = T.cast(T.cast(new_res_local[i_mhc, i1_h], T.bfloat16), T.float32)

                if L > L_post:
                    pre_mix_tensor_last = T.make_tensor(pre_mix_ptrs[L_post], (n, mhc), T.float32)
                    layer_input_tensor_last = T.make_tensor(layer_input_ptrs[L_post], (n, h), T.bfloat16)

                    T.copy(pre_mix_tensor_last[i_n, 0], pre_mix_local)

                    T.clear(layer_input_local)
                    for i_mhc in T.serial(mhc):
                        for i1_h in T.Parallel(h_blk):
                            layer_input_local[i1_h] += pre_mix_local[i_mhc] * res_local[i_mhc, i1_h]

                    T.copy(layer_input_local, layer_input_tensor_last[i_n, i0_h * h_blk])

    return kernel


def mhc_multilayer_recompute(
    initial_residual: torch.Tensor,
    pre_mix_list: list[torch.Tensor],
    layer_output_list: list[torch.Tensor],
    post_mix_list: list[torch.Tensor],
    comb_mix_list: list[torch.Tensor],
    layer_input_list: list[torch.Tensor],
    residual_list: list[torch.Tensor],
) -> None:
    num_layers = len(pre_mix_list)
    num_post = len(layer_output_list)
    assert num_layers == len(layer_input_list)
    assert num_post == len(post_mix_list) == len(comb_mix_list) == len(residual_list)
    assert num_post == num_layers - 1 or num_post == num_layers, (
        f'post count ({num_post}) must be num_layers-1 or num_layers (num_layers={num_layers})'
    )
    assert num_layers > 0

    mhc_mult = initial_residual.shape[-2]
    hidden = initial_residual.shape[-1]

    (
        pre_mix_ptrs,
        layer_input_ptrs,
        residual_ptrs,
        layer_output_ptrs,
        post_mix_ptrs,
        comb_mix_ptrs,
    ) = _make_ptr_tables_batched(
        [pre_mix_list, layer_input_list, residual_list, layer_output_list, post_mix_list, comb_mix_list],
        device=initial_residual.device,
    )

    # Two Wave64 groups improve the 4096/7168 hidden shapes; 2560 and 8192 retain one.
    n_thr = 128 if hidden in (4096, 7168) else 64
    # 8192 needs smaller hidden tiles to avoid its large per-CTA fragment footprint.
    h_blk = 512 if hidden == 8192 else 2048
    kernel = _mhc_multilayer_recompute_kernel(
        mhc_mult,
        hidden,
        num_layers,
        num_post,
        n_thr=n_thr,
        h_blk=h_blk,
    )
    kernel(
        initial_residual.view(-1, mhc_mult, hidden),
        pre_mix_ptrs,
        layer_output_ptrs,
        post_mix_ptrs,
        comb_mix_ptrs,
        layer_input_ptrs,
        residual_ptrs,
    )
