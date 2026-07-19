import torch

from tile_kernels.mhc.pre_apply_mix_kernel import _mhc_pre_apply_mix_bwd, _mhc_pre_apply_mix_fwd


class MHCPreApplyMix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: 'MHCPreApplyMix',
        x: torch.Tensor,
        mix: torch.Tensor,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, mix)
        h = x.shape[-1]
        mhc = mix.shape[-2]
        assert mix.shape[-1] == 1
        ctx.fwd_kernel = _mhc_pre_apply_mix_fwd(mhc, h)
        # Backward shared-memory pressure has a different optimum than the forward kernel.
        bwd_n_thr = 256 if h in (2560, 4096) else 128
        bwd_h_blk = 512 if h in (2560, 4096, 7168) else 1024
        ctx.bwd_kernel = _mhc_pre_apply_mix_bwd(
            mhc,
            h,
            n_thr=bwd_n_thr,
            h_blk=bwd_h_blk,
        )
        if out is None:
            out = torch.empty(*x.shape[:-2], h, dtype=torch.bfloat16, device=x.device)
        ctx.fwd_kernel(x.view(-1, mhc, h), mix.view(-1, mhc), out.view(-1, h))
        return out

    @staticmethod
    def backward(ctx: 'MHCPreApplyMix', o_grad: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor, None]:
        x, mix = ctx.saved_tensors
        h = x.shape[-1]
        mhc = mix.shape[-2]
        if hasattr(x.untyped_storage(), 'grad_from_mhc_post'):
            x_grad = x.untyped_storage().grad_from_mhc_post
            mix_grad = ctx.bwd_kernel(
                o_grad.view(-1, h),
                x.view(-1, mhc, h),
                mix.view(-1, mhc),
                x_grad.view(-1, mhc, h),
            )
            x_grad = None
        else:
            x_grad = torch.zeros_like(x)
            zero_grad_bwd_kernel = _mhc_pre_apply_mix_bwd(
                mhc,
                h,
                n_thr=256 if h in (2560, 4096) else 128,
                h_blk=512 if h in (2560, 4096, 7168) else 1024,
                x_grad_is_zero=True,
            )
            mix_grad = zero_grad_bwd_kernel(
                o_grad.view(-1, h),
                x.view(-1, mhc, h),
                mix.view(-1, mhc),
                x_grad.view(-1, mhc, h),
            )
        return x_grad, mix_grad.view_as(mix), None


def mhc_pre_apply_mix(
    x: torch.Tensor,
    mix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return MHCPreApplyMix.apply(x, mix, out)
