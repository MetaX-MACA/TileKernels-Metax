import torch

from tile_kernels.mhc.expand_kernel import expand_to_mhc_bwd_tl, expand_to_mhc_fwd_tl


class ExpandToMHCFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: 'ExpandToMHCFn',
        hidden: torch.Tensor,
        mhc_mult: int,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        if out is None:
            out = hidden.new_empty(*hidden.shape[:-1], mhc_mult, hidden.shape[-1])
        assert hidden.is_contiguous()
        hidden_flat = hidden.flatten(0, -2)
        num_tokens = hidden_flat.shape[0]
        use_blk_n64 = num_tokens % 64 == 0 and not (
            num_tokens >= 8192 and hidden.shape[-1] == 1280
        )
        use_blk_h256 = use_blk_n64 and hidden.shape[-1] >= 4096
        kernel = expand_to_mhc_fwd_tl(
            hidden.shape[-1],
            mhc_mult,
            blk_n=64 if use_blk_n64 else 32,
            blk_h=256 if use_blk_h256 else 128,
        )
        kernel(hidden_flat, out.flatten(0, -3))
        return out

    @staticmethod
    def backward(ctx: 'ExpandToMHCFn', out_grad: torch.Tensor) -> torch.Tensor:
        hidden_grad = out_grad.new_empty(*out_grad.shape[:-2], out_grad.shape[-1])
        kernel = expand_to_mhc_bwd_tl(out_grad.shape[-1], out_grad.shape[-2])
        kernel(out_grad.flatten(0, -3), hidden_grad.flatten(0, -2))
        return hidden_grad, None, None


def expand_to_mhc(
    hidden: torch.Tensor,
    mhc_mult: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return ExpandToMHCFn.apply(hidden, mhc_mult, out)
