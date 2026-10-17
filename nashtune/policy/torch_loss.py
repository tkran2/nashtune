"""The GRPO/SFT loss in torch. A line-for-line mirror of
``nashtune.grpo.loss_and_grad``; tests/test_torch_backend.py checks that the
two give the same loss and gradient."""

from __future__ import annotations

import torch

from ..grpo import UpdateBatch


def loss_from_logp_torch(logps: list["torch.Tensor"], batch: UpdateBatch) -> "torch.Tensor":
    eps = float(batch.clip_eps)
    total = None
    for j, lp in enumerate(logps):

        def t(x):
            return torch.as_tensor(x, dtype=lp.dtype, device=lp.device)

        ratio = torch.exp(lp - t(batch.old_logp[j]))
        clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        surrogate = (t(batch.adv_pos[j]) * torch.minimum(ratio, clipped)
                     + t(batch.adv_neg[j]) * torch.maximum(ratio, clipped)).sum()
        loss = -surrogate
        kl_coef = float(batch.kl_coef[j])
        if kl_coef != 0.0:
            p = torch.exp(lp)
            loss = loss + kl_coef * (p * (lp - t(batch.ref_logp[j]))).sum()
        ce_weight = float(batch.ce_weight[j])
        if ce_weight != 0.0:
            loss = loss - ce_weight * (t(batch.ce_target[j]) * lp).sum()
        total = loss if total is None else total + loss
    if total is None:
        raise ValueError("empty batch")
    return total
