################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Run learned allocations through deterministic capacity-aware deployment."""

from __future__ import annotations

import numpy as np
import torch

from .windows import WindowBatch


def round_allocation(
    probabilities: np.ndarray,
    demands: np.ndarray,
    capacities: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    threshold: float = 1.0e-4,
) -> np.ndarray:
    """Greedily assign candidate cells under job and platform constraints."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [jobs, candidates]")
    if demands.shape[:2] != probabilities.shape:
        raise ValueError("demands must match the probability grid")
    if candidate_mask.shape != probabilities.shape:
        raise ValueError("candidate_mask must match the probability grid")
    if demands.shape[2:] != capacities.shape:
        raise ValueError("demand platforms must match capacities")

    n_jobs, n_candidates = probabilities.shape
    remaining = capacities.astype(float).copy()
    assignment = np.full(n_jobs, -1, dtype=np.int64)
    order = np.argsort(-probabilities, axis=None, kind="stable")
    for flat_index in order:
        job_index, candidate_index = divmod(
            int(flat_index),
            n_candidates,
        )
        if assignment[job_index] >= 0:
            continue
        if not candidate_mask[job_index, candidate_index]:
            continue
        if probabilities[job_index, candidate_index] < threshold:
            continue
        demand = demands[job_index, candidate_index]
        if np.all(remaining - demand >= -1.0e-9):
            assignment[job_index] = candidate_index
            remaining -= demand
    return assignment


@torch.no_grad()
def deployed_outcome(
    net: torch.nn.Module,
    batch: WindowBatch,
    values: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return greedy assignments and individually rational payments."""
    if values.shape != batch.true_values.shape:
        raise ValueError(
            f"values must have shape {tuple(batch.true_values.shape)}"
        )
    net.eval()
    probabilities, payment_fractions = net(
        batch.input_tensor(values),
        batch.slot_mask,
        batch.job_mask,
    )
    probabilities_array = probabilities.detach().cpu().numpy()
    demands = batch.demands.detach().cpu().numpy()
    capacities = batch.capacities.detach().cpu().numpy()
    costs = (
        batch.public_channels[..., 0] * batch.scales.unsqueeze(-1)
    ).detach().cpu().numpy()
    reported_values = values.detach().cpu().numpy()
    valid = batch.candidate_mask.detach().cpu().numpy()
    valid = valid & (reported_values + 1.0e-9 >= costs)

    batch_size, n_jobs = values.shape[:2]
    assignments = np.full((batch_size, n_jobs), -1, dtype=np.int64)
    payments = np.zeros((batch_size, n_jobs), dtype=float)
    fractions = payment_fractions.detach().cpu().numpy()
    for batch_index in range(batch_size):
        assignments[batch_index] = round_allocation(
            probabilities_array[batch_index],
            demands[batch_index],
            capacities[batch_index],
            valid[batch_index],
        )
        for job_index, candidate_index in enumerate(
            assignments[batch_index]
        ):
            if candidate_index < 0:
                continue
            value = float(
                reported_values[batch_index, job_index, candidate_index]
            )
            cost = float(costs[batch_index, job_index, candidate_index])
            fraction_charge = float(fractions[batch_index, job_index]) * value
            payments[batch_index, job_index] = min(
                value,
                max(cost, fraction_charge),
            )
    return assignments, payments
