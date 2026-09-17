################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Permutation-aware neural layers used by RegretFormer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dimension: int,
) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    total = (values * weights).sum(dim=dimension, keepdim=True)
    count = weights.sum(dim=dimension, keepdim=True).clamp_min(1.0e-9)
    return total / count


class Exchangeable(nn.Module):
    """Apply an equivariant linear layer to a masked job-candidate grid."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create channel, job-pool, candidate-pool, and global projections."""
        super().__init__()
        self.channel_layer = nn.Linear(in_channels, out_channels)
        self.job_pool_layer = nn.Linear(
            in_channels,
            out_channels,
            bias=False,
        )
        self.candidate_pool_layer = nn.Linear(
            in_channels,
            out_channels,
            bias=False,
        )
        self.global_pool_layer = nn.Linear(
            in_channels,
            out_channels,
            bias=False,
        )

    def forward(
        self,
        values: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Transform ``[batch, jobs, candidates, channels]`` values."""
        job_pool = _masked_mean(values, slot_mask, 1)
        candidate_pool = _masked_mean(values, slot_mask, 2)
        job_mask = slot_mask.any(dim=2)
        global_pool = _masked_mean(candidate_pool, job_mask, 1)
        return (
            self.channel_layer(values)
            + self.job_pool_layer(job_pool)
            + self.candidate_pool_layer(candidate_pool)
            + self.global_pool_layer(global_pool)
        )


class ScaledDotProductAttention(nn.Module):
    """Compute scaled dot-product attention with an optional key mask."""

    def __init__(self, temperature: float) -> None:
        """Set the positive scale applied to query vectors."""
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attended values and attention weights."""
        attention = torch.matmul(
            query / self.temperature,
            key.transpose(-2, -1),
        )
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1.0e9)
        attention = functional.softmax(attention, dim=-1)
        return torch.matmul(attention, value), attention


class MultiHeadAttention(nn.Module):
    """Apply pre-normalized multi-head self-attention with a residual."""

    def __init__(self, n_heads: int, model_width: int, head_width: int) -> None:
        """Create attention projections for the requested dimensions."""
        super().__init__()
        if min(n_heads, model_width, head_width) < 1:
            raise ValueError("attention dimensions must be positive")
        self.n_heads = n_heads
        self.head_width = head_width
        projection_width = n_heads * head_width
        self.query = nn.Linear(model_width, projection_width, bias=False)
        self.key = nn.Linear(model_width, projection_width, bias=False)
        self.value = nn.Linear(model_width, projection_width, bias=False)
        self.output = nn.Linear(projection_width, model_width, bias=False)
        self.attention = ScaledDotProductAttention(head_width**0.5)
        self.layer_norm = nn.LayerNorm(model_width, eps=1.0e-6)

    def forward(
        self,
        values: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over the sequence dimension of ``values``."""
        batch_size, sequence_length = values.shape[:2]
        residual = values
        normalized = self.layer_norm(values)

        def project(layer: nn.Linear) -> torch.Tensor:
            projected = layer(normalized).view(
                batch_size,
                sequence_length,
                self.n_heads,
                self.head_width,
            )
            return projected.transpose(1, 2)

        mask = None
        if key_mask is not None:
            mask = key_mask[:, None, None, :]
        attended, _ = self.attention(
            project(self.query),
            project(self.key),
            project(self.value),
            mask,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            -1,
        )
        return self.output(attended) + residual


class DualAxisAttentionBlock(nn.Module):
    """Attend over candidates within jobs, then jobs at each candidate rank."""

    def __init__(self, n_heads: int, hid: int, hid_att: int) -> None:
        """Create both attention axes and their fusion projection."""
        super().__init__()
        self.candidate_attention = MultiHeadAttention(
            n_heads,
            hid,
            hid_att,
        )
        self.job_attention = MultiHeadAttention(
            n_heads,
            hid,
            hid_att,
        )
        self.fusion = nn.Linear(2 * hid, hid)

    def forward(
        self,
        values: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Transform a masked ``[batch, jobs, candidates, hidden]`` grid."""
        batch_size, n_jobs, n_candidates, hidden = values.shape
        by_candidate = values.reshape(
            batch_size * n_jobs,
            n_candidates,
            hidden,
        )
        by_candidate = self.candidate_attention(
            by_candidate,
            slot_mask.reshape(batch_size * n_jobs, n_candidates),
        ).reshape(values.shape)

        by_job = values.permute(0, 2, 1, 3).reshape(
            batch_size * n_candidates,
            n_jobs,
            hidden,
        )
        job_key_mask = slot_mask.permute(0, 2, 1).reshape(
            batch_size * n_candidates,
            n_jobs,
        )
        empty_rows = ~job_key_mask.any(dim=1)
        if bool(empty_rows.any()):
            job_key_mask = job_key_mask.clone()
            job_key_mask[empty_rows, 0] = True
        by_job = self.job_attention(by_job, job_key_mask)
        by_job = by_job.reshape(
            batch_size,
            n_candidates,
            n_jobs,
            hidden,
        ).permute(0, 2, 1, 3)

        fused = torch.tanh(torch.cat((by_candidate, by_job), dim=-1))
        transformed = self.fusion(fused) + values
        return transformed * slot_mask.unsqueeze(-1).to(values.dtype)


class AllocationPaymentHead(nn.Module):
    """Produce candidate probabilities and one payment fraction per job.

    Probabilities are over the candidates. The reject probability is one minus
    their sum.
    """

    def __init__(self, hid: int) -> None:
        """Create allocation, rejection, and payment projections."""
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid, eps=1.0e-6)
        self.allocation = nn.Sequential(
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, 1),
        )
        self.rejection = nn.Sequential(
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, 1),
        )
        self.payment = nn.Sequential(
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        values: torch.Tensor,
        slot_mask: torch.Tensor,
        job_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return allocation probabilities and payment fractions."""
        normalized = self.layer_norm(values)
        logits = self.allocation(normalized).squeeze(-1)
        logits = logits.masked_fill(~slot_mask, -1.0e9)
        job_embedding = _masked_mean(normalized, slot_mask, 2).squeeze(2)
        rejection = self.rejection(job_embedding)
        probabilities = functional.softmax(
            torch.cat((logits, rejection), dim=-1),
            dim=-1,
        )[..., :-1]
        probabilities = (
            probabilities
            * slot_mask.to(probabilities.dtype)
            * job_mask.unsqueeze(-1).to(probabilities.dtype)
        )
        payment = self.payment(job_embedding).squeeze(-1)
        payment = payment * job_mask.to(payment.dtype)
        return probabilities, payment


class RegretFormerNet(nn.Module):
    """Map six-channel window tensors to allocations and payment fractions."""

    def __init__(
        self,
        in_channels: int = 6,
        hid: int = 64,
        hid_att: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        """Build the exchangeable embedding, attention body, and output head."""
        super().__init__()
        if min(in_channels, hid, hid_att, n_layers, n_heads) < 1:
            raise ValueError("network dimensions must be positive")
        self.input_layer = Exchangeable(in_channels, hid)
        self.body = nn.ModuleList(
            DualAxisAttentionBlock(n_heads, hid, hid_att)
            for _ in range(n_layers)
        )
        self.head = AllocationPaymentHead(hid)

    def forward(
        self,
        values: torch.Tensor,
        slot_mask: torch.Tensor,
        job_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked allocation probabilities and payment fractions."""
        mask = slot_mask.unsqueeze(-1).to(values.dtype)
        hidden = torch.tanh(self.input_layer(values, slot_mask)) * mask
        for block in self.body:
            hidden = block(hidden, slot_mask)
        return self.head(hidden, slot_mask, job_mask)
