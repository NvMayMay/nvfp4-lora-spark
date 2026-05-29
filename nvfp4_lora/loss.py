"""Memory-oriented loss helpers for NVFP4 LoRA training."""
import torch
import torch.nn.functional as F

try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    _LIGER_AVAILABLE = True
except ImportError:
    LigerFusedLinearCrossEntropyFunction = None
    _LIGER_AVAILABLE = False


class _ChunkedFrozenLMHeadCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, labels, lm_head_weight, chunk_tokens, logits_fp32):
        if lm_head_weight.requires_grad:
            raise RuntimeError("chunked frozen CE requires a frozen lm_head weight")
        if chunk_tokens < 1:
            raise RuntimeError("chunk_tokens must be >= 1")

        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:].contiguous()
        valid_count = int((shift_labels != -100).sum().item())
        denom = max(1, valid_count)

        total_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, shift_hidden.shape[1], chunk_tokens):
                end = min(start + chunk_tokens, shift_hidden.shape[1])
                logits = F.linear(shift_hidden[:, start:end, :].to(lm_head_weight.dtype), lm_head_weight)
                if logits_fp32:
                    logits = logits.float()
                total_loss = total_loss + F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    shift_labels[:, start:end].reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

        ctx.save_for_backward(hidden_states, labels, lm_head_weight)
        ctx.chunk_tokens = int(chunk_tokens)
        ctx.denom = denom
        ctx.logits_fp32 = bool(logits_fp32)
        return total_loss / denom

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, labels, lm_head_weight = ctx.saved_tensors
        chunk_tokens = ctx.chunk_tokens
        denom = ctx.denom
        logits_fp32 = ctx.logits_fp32

        shift_labels = labels[:, 1:].contiguous()
        scale = grad_output.to(torch.float32)
        grad_hidden = None

        with torch.enable_grad():
            for start in range(0, hidden_states.shape[1] - 1, chunk_tokens):
                end = min(start + chunk_tokens, hidden_states.shape[1] - 1)
                hidden_chunk = hidden_states[:, start:end, :].detach().requires_grad_(True)
                logits = F.linear(hidden_chunk.to(lm_head_weight.dtype), lm_head_weight)
                if logits_fp32:
                    logits = logits.float()
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    shift_labels[:, start:end].reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ) / denom
                (grad_chunk,) = torch.autograd.grad(loss, hidden_chunk)
                if grad_hidden is None:
                    grad_hidden = torch.empty_like(hidden_states)
                    if hidden_states.shape[1] > 0:
                        grad_hidden[:, -1:, :].zero_()
                # NOTE: scale is downcast to grad_chunk.dtype before the multiply; the pre-patch expression `grad_chunk * scale` promoted to fp32. Currently safe because callers pass grad_output=1.0.
                torch.mul(grad_chunk, scale.to(grad_chunk.dtype), out=grad_hidden[:, start:end, :])

        if grad_hidden is None:
            grad_hidden = torch.zeros_like(hidden_states)

        return grad_hidden, None, None, None, None


def chunked_frozen_lm_head_ce(hidden_states, labels, lm_head, chunk_tokens, logits_fp32=True):
    """Cross entropy for a frozen lm_head without retaining full-sequence logits."""
    return _ChunkedFrozenLMHeadCE.apply(
        hidden_states, labels, lm_head.weight, int(chunk_tokens), bool(logits_fp32)
    )


def liger_fused_lm_head_ce(hidden_states, labels, lm_head, ignore_index=-100, accum_dtype=None):
    """Liger FusedLinearCrossEntropy wrapper for a frozen lm_head.

    Avoids materializing the (seq, vocab) logits tensor entirely; computes
    per-row grad-of-hidden-states inside a single fused triton kernel. Equivalent
    in math to chunked_frozen_lm_head_ce with `reduction='mean'` over non-ignored
    tokens, but with substantially fewer CUDA allocation events on the backward
    path (the main reason to use it on the GB10 descriptor-cliff path).
    """
    if not _LIGER_AVAILABLE:
        raise RuntimeError(
            "--loss-mode liger_flce requires liger-kernel; "
            "install with `pip install liger-kernel` in the same venv."
        )
    if lm_head.weight.requires_grad:
        raise RuntimeError("Liger fused frozen CE requires a frozen lm_head weight")

    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat_hidden = shift_hidden.reshape(-1, shift_hidden.shape[-1])
    flat_labels = shift_labels.reshape(-1)

    valid_count = int((flat_labels != ignore_index).sum().item())
    if valid_count == 0:
        # No supervised tokens — return a zero scalar that preserves the autograd link to
        # hidden_states so the training loop's loss.backward() does not crash with
        # "element 0 of tensors does not require grad". The chunked path avoids this via
        # _ChunkedFrozenLMHeadCE.apply()'s implicit grad_fn; here we have to construct it.
        return (flat_hidden.float() * 0.0).sum()

    outputs = LigerFusedLinearCrossEntropyFunction.apply(
        flat_hidden,
        lm_head.weight,
        flat_labels,
        None,           # bias
        None,           # ce_weight
        ignore_index,
        0.0,            # lse_square_scale
        0.0,            # label_smoothing
        "mean",         # reduction
        None,           # softcap
        False,          # return_z_loss
        accum_dtype,    # accum_dtype
        False,          # use_token_scaling
        False,          # return_token_accuracy
        False,          # return_predicted_tokens
    )
    # LigerFLCE returns (loss, z_loss, token_accuracy, predicted_tokens); take loss only.
    return outputs[0]
