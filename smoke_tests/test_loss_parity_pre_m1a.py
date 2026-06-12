#!/usr/bin/env python3
"""Pre-M1a deterministic loss/gradient regression gate.

This test intentionally does not import the old loss.py beside the refactored
one; both implementations cannot safely occupy the same process. The expected
(loss, grad_norm) constants below are fixed CPU values for the refactored path,
recorded during test authoring, so future loss-path edits have a cheap
regression tripwire.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import nvfp4_lora.loss as loss_mod
from nvfp4_lora.loss import chunked_frozen_lm_head_ce, liger_fused_lm_head_ce


N_TOKENS = 64
LOG2_F32 = 0.6931471824645996

EXPECTED = {
    0: {
        1.0: (0.0, 0.0),
        0.5: (0.0, 0.0),
    },
    1: {
        1.0: (LOG2_F32, 1.0),
        0.5: (LOG2_F32, 0.5),
    },
    32: {
        1.0: (LOG2_F32, 0.17677669529663687),
        0.5: (LOG2_F32, 0.08838834764831845),
    },
    64: {
        1.0: (LOG2_F32, 0.125),
        0.5: (LOG2_F32, 0.0625),
    },
}


class _CpuLigerFusedLinearCrossEntropyFunction:
    @staticmethod
    def apply(
        flat_hidden,
        weight,
        flat_labels,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        accum_dtype,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
    ):
        logits = F.linear(flat_hidden, weight, bias)
        loss = F.cross_entropy(
            logits,
            flat_labels,
            weight=ce_weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        return loss, None, None, None


def _install_cpu_liger_stub():
    loss_mod._LIGER_AVAILABLE = True
    loss_mod.LigerFusedLinearCrossEntropyFunction = _CpuLigerFusedLinearCrossEntropyFunction


def _make_lm_head():
    lm_head = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.tensor([[0.0], [2.0]], dtype=torch.float32))
    lm_head.weight.requires_grad_(False)
    return lm_head


def _make_case(valid_count):
    torch.manual_seed(20260531)
    flat_hidden = torch.zeros(N_TOKENS, 1, dtype=torch.float32, requires_grad=True)
    tail_hidden = torch.zeros(1, 1, 1, dtype=torch.float32)
    hidden_states = torch.cat([flat_hidden.view(1, N_TOKENS, 1), tail_hidden], dim=1)

    labels = torch.full((1, N_TOKENS + 1), -100, dtype=torch.long)
    label_pattern = torch.arange(N_TOKENS, dtype=torch.long) % 2
    if valid_count > 0:
        labels[0, 1 : 1 + valid_count] = label_pattern[:valid_count]
    return flat_hidden, hidden_states, labels


def _compute_loss(loss_name, hidden_states, labels, lm_head, chunk_tokens=8):
    if loss_name == "chunked":
        return chunked_frozen_lm_head_ce(
            hidden_states,
            labels,
            lm_head,
            chunk_tokens=chunk_tokens,
            logits_fp32=True,
        )
    if loss_name == "liger":
        _install_cpu_liger_stub()
        return liger_fused_lm_head_ce(hidden_states, labels, lm_head)
    raise AssertionError(f"unknown loss_name: {loss_name}")


@pytest.mark.parametrize("loss_name", ["chunked", "liger"])
@pytest.mark.parametrize("valid_count", [0, 1, 32, 64])
@pytest.mark.parametrize("grad_output", [1.0, 0.5])
def test_loss_and_hidden_grad_constants(loss_name, valid_count, grad_output):
    lm_head = _make_lm_head()
    flat_hidden, hidden_states, labels = _make_case(valid_count)

    loss = _compute_loss(loss_name, hidden_states, labels, lm_head)
    (grad,) = torch.autograd.grad(
        loss,
        flat_hidden,
        grad_outputs=torch.tensor(grad_output, dtype=loss.dtype),
    )

    expected_loss, expected_grad_norm = EXPECTED[valid_count][grad_output]
    assert torch.allclose(
        loss.detach(),
        torch.tensor(expected_loss, dtype=loss.dtype),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        grad.norm().detach(),
        torch.tensor(expected_grad_norm, dtype=grad.dtype),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("loss_name", ["chunked", "liger"])
def test_all_ignored_chunk_next_to_normal_chunk_has_no_nan_grad(loss_name):
    torch.manual_seed(20260531)
    lm_head = _make_lm_head()

    flat_hidden = torch.randn(8, 1, dtype=torch.float32, requires_grad=True)
    hidden_states = torch.cat([flat_hidden.view(1, 8, 1), torch.zeros(1, 1, 1)], dim=1)

    labels = torch.full((1, 9), -100, dtype=torch.long)
    labels[0, 5:9] = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    loss = _compute_loss(loss_name, hidden_states, labels, lm_head, chunk_tokens=4)
    (grad,) = torch.autograd.grad(loss, flat_hidden)

    assert not torch.isnan(grad).any()
    assert math.isfinite(float(loss.detach()))