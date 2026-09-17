################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Estimate deployed RegretFormer regret with unilateral report attacks.

The grid and guided-ascent procedures follow You et al. (2026). Both report a
lower bound on true regret because they search only finitely many misreports.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import numpy as np
import torch

from .deploy import deployed_outcome
from .windows import WindowBatch

DEFAULT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


def _grid_values(grid: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in grid)
    if not values:
        raise ValueError("grid must not be empty")
    if any(not isfinite(value) or value < 0.0 for value in values):
        raise ValueError("grid factors must be finite and non-negative")
    return values


def _validate_dimension_values(
    batch: WindowBatch,
    dimension_values: torch.Tensor,
) -> None:
    if dimension_values.shape != batch.dimension_values.shape:
        raise ValueError(
            "true_values must have shape "
            f"{tuple(batch.dimension_values.shape)}"
        )


def _deployed_utilities(
    net: torch.nn.Module,
    batch: WindowBatch,
    reported_dimensions: torch.Tensor,
    true_dimensions: torch.Tensor,
) -> torch.Tensor:
    reported_values = batch.candidate_values(reported_dimensions)
    true_values = batch.candidate_values(true_dimensions)
    assignments, payments = deployed_outcome(net, batch, reported_values)
    utilities = torch.zeros_like(batch.job_mask, dtype=true_values.dtype)
    for batch_index, row in enumerate(assignments):
        for job_index, candidate_index in enumerate(row):
            if candidate_index >= 0:
                utilities[batch_index, job_index] = (
                    true_values[batch_index, job_index, candidate_index]
                    - float(payments[batch_index, job_index])
                )
    return utilities


def _unilateral_dimensions(
    true_values: torch.Tensor,
    batch_index: int,
    job_index: int,
    factors: torch.Tensor,
) -> torch.Tensor:
    selector = torch.zeros_like(true_values)
    selector[batch_index, job_index] = 1.0
    factor_grid = torch.ones_like(true_values)
    factor_grid = factor_grid + selector * (factors - 1.0)
    return true_values * factor_grid


def grid_regret(
    net: torch.nn.Module,
    batch: WindowBatch,
    true_values: torch.Tensor,
    grid: Sequence[float] = DEFAULT_GRID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deployed grid regret and its per-dimension report factors.

    Each valid report dimension is scaled alone, then the whole report vector
    is scaled jointly, while every other job reports truthfully. The result is
    a finite-search lower bound on true regret.
    """
    _validate_dimension_values(batch, true_values)
    grid_values = _grid_values(grid)
    baseline = _deployed_utilities(net, batch, true_values, true_values)
    regret = torch.zeros_like(baseline)
    argmax = torch.ones_like(true_values)

    for batch_index in range(true_values.shape[0]):
        for job_index in range(true_values.shape[1]):
            if not bool(batch.job_mask[batch_index, job_index]):
                continue
            valid = batch.dimension_mask[batch_index, job_index]
            dimension_indices = torch.nonzero(valid).flatten().tolist()
            factor_vectors = []
            for dimension_index in dimension_indices:
                for factor in grid_values:
                    factors = torch.ones_like(
                        true_values[batch_index, job_index]
                    )
                    factors[dimension_index] = factor
                    factor_vectors.append(factors)
            for factor in grid_values:
                factors = torch.ones_like(true_values[batch_index, job_index])
                factors[valid] = factor
                factor_vectors.append(factors)

            for factors in factor_vectors:
                reported = _unilateral_dimensions(
                    true_values,
                    batch_index,
                    job_index,
                    factors,
                )
                utility = _deployed_utilities(
                    net,
                    batch,
                    reported,
                    true_values,
                )[batch_index, job_index]
                gain = torch.clamp(
                    utility - baseline[batch_index, job_index],
                    min=0.0,
                )
                if float(gain) > float(regret[batch_index, job_index]):
                    regret[batch_index, job_index] = gain
                    argmax[batch_index, job_index] = factors
    return regret, argmax


def _relaxed_utility(
    net: torch.nn.Module,
    batch: WindowBatch,
    reported_dimensions: torch.Tensor,
    true_dimensions: torch.Tensor,
    batch_index: int,
    job_index: int,
) -> torch.Tensor:
    reported_values = batch.candidate_values(reported_dimensions)
    true_values = batch.candidate_values(true_dimensions)
    probabilities, payment_fractions = net(
        batch.input_tensor(reported_values),
        batch.slot_mask,
        batch.job_mask,
    )
    costs = batch.public_channels[..., 0] * batch.scales.unsqueeze(-1)
    fraction_charge = (
        payment_fractions.unsqueeze(-1) * reported_values
    )
    charges = torch.minimum(
        reported_values,
        torch.maximum(costs, fraction_charge),
    )
    utility_by_candidate = true_values - charges
    return (
        probabilities[batch_index, job_index]
        * utility_by_candidate[batch_index, job_index]
    ).sum()


def guided_refinement_regret(
    net: torch.nn.Module,
    batch: WindowBatch,
    true_values: torch.Tensor,
    *,
    grid: Sequence[float] = DEFAULT_GRID,
    steps: int = 100,
    lr: float = 0.05,
    n_perturb: int = 8,
    sigma: float = 0.6,
    seed: int = 0,
) -> torch.Tensor:
    """Refine grid attacks by ascent and deployed rescoring.

    Starts include the grid argmax, perturbations around it and truth, and
    uniform random factors. Every refined report is scored through deployment.
    The returned estimate remains a lower bound on true regret.
    """
    _validate_dimension_values(batch, true_values)
    grid_values = _grid_values(grid)
    if steps < 0 or n_perturb < 0:
        raise ValueError("steps and n_perturb must be non-negative")
    if lr <= 0.0 or sigma < 0.0:
        raise ValueError("lr must be positive and sigma non-negative")
    generator = np.random.default_rng(seed)
    grid_bound, grid_argmax = grid_regret(
        net,
        batch,
        true_values,
        grid_values,
    )
    baseline = _deployed_utilities(net, batch, true_values, true_values)
    result = grid_bound.clone()
    low = min(grid_values)
    high = max(grid_values)
    net.eval()

    for batch_index in range(true_values.shape[0]):
        for job_index in range(true_values.shape[1]):
            if not bool(batch.job_mask[batch_index, job_index]):
                continue
            valid = batch.dimension_mask[batch_index, job_index]
            grid_start = grid_argmax[batch_index, job_index].cpu().numpy()
            starts = [grid_start]
            for _ in range(n_perturb):
                starts.extend((
                    np.clip(
                        grid_start + generator.normal(0.0, sigma, grid_start.shape),
                        low,
                        high,
                    ),
                    np.clip(
                        1.0 + generator.normal(0.0, sigma, grid_start.shape),
                        low,
                        high,
                    ),
                    generator.uniform(low, high, grid_start.shape),
                ))

            for start in starts:
                factors = torch.tensor(
                    start,
                    dtype=true_values.dtype,
                    device=true_values.device,
                    requires_grad=True,
                )
                optimizer = torch.optim.Adam((factors,), lr=lr)
                for _ in range(steps):
                    optimizer.zero_grad()
                    clamped = torch.where(
                        valid,
                        factors.clamp(low, high),
                        torch.ones_like(factors),
                    )
                    reported = _unilateral_dimensions(
                        true_values,
                        batch_index,
                        job_index,
                        clamped,
                    )
                    utility = _relaxed_utility(
                        net,
                        batch,
                        reported,
                        true_values,
                        batch_index,
                        job_index,
                    )
                    gradient = torch.autograd.grad(-utility, factors)[0]
                    factors.grad = gradient
                    optimizer.step()

                final_factors = torch.where(
                    valid,
                    factors.detach().clamp(low, high),
                    torch.ones_like(factors),
                )
                reported = _unilateral_dimensions(
                    true_values,
                    batch_index,
                    job_index,
                    final_factors,
                )
                utility = _deployed_utilities(
                    net,
                    batch,
                    reported,
                    true_values,
                )[batch_index, job_index]
                gain = torch.clamp(
                    utility - baseline[batch_index, job_index],
                    min=0.0,
                )
                result[batch_index, job_index] = torch.maximum(
                    result[batch_index, job_index],
                    gain,
                )
    return result


def summarize_regret(
    regret: torch.Tensor,
    batch: WindowBatch,
    true_values: torch.Tensor,
    net: torch.nn.Module,
) -> dict[str, float]:
    """Summarize regret and normalize it by deployed truthful utility."""
    _validate_dimension_values(batch, true_values)
    if regret.shape != batch.job_mask.shape:
        raise ValueError(f"regret must have shape {tuple(batch.job_mask.shape)}")
    truthful = _deployed_utilities(net, batch, true_values, true_values)
    mask = batch.job_mask
    selected_regret = regret[mask].detach().cpu().numpy()
    selected_utility = truthful[mask].detach().cpu().numpy()
    if not selected_regret.size:
        return {
            "mean_regret": 0.0,
            "median_regret": 0.0,
            "p95_regret": 0.0,
            "max_regret": 0.0,
            "mean_regret_over_mean_truthful_utility": 0.0,
        }
    mean_regret = float(np.mean(selected_regret))
    mean_utility = float(np.mean(selected_utility))
    return {
        "mean_regret": mean_regret,
        "median_regret": float(np.median(selected_regret)),
        "p95_regret": float(np.percentile(selected_regret, 95)),
        "max_regret": float(np.max(selected_regret)),
        "mean_regret_over_mean_truthful_utility": (
            mean_regret / max(mean_utility, 1.0e-9)
        ),
    }
