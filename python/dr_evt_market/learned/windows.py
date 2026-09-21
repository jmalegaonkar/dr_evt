################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Turn windows into padded tensors for the learned mechanism.

A job's report is its multipliers: one for a single bid, one per platform for
a per-platform bid. Each candidate's value is a fixed linear function of those
multipliers, ``sum_d weight[candidate, d] * multiplier[d]``, where the weight
is the job's base cost for a single bid and the cost of the legs on that
platform for a per-platform bid. Misreports scale multipliers, so every
misreport is a report the job could have made.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log

import numpy as np
import torch

from ..mechanisms.base import Window, demand

PUBLIC_CHANNELS = 5
IN_CHANNELS = PUBLIC_CHANNELS + 1
SINGLE = "*"


def _read_only(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class ValueSamplingSpec:
    """Configure synthetic multipliers: a lognormal factor times a preference."""

    lognormal_mean: float = log(2.0)
    lognormal_sigma: float = 0.5
    preference_low: float = 0.75
    preference_high: float = 1.25

    def __post_init__(self) -> None:
        if self.lognormal_sigma < 0.0:
            raise ValueError("lognormal_sigma must be non-negative")
        if self.preference_low < 0.0:
            raise ValueError("preference_low must be non-negative")
        if self.preference_high < self.preference_low:
            raise ValueError("preference_high must be at least preference_low")


@dataclass(frozen=True)
class WindowStructure:
    """One window as ragged, immutable arrays."""

    job_ids: tuple[str, ...]
    candidate_ids: tuple[tuple[str, ...], ...]
    public_channels: tuple[np.ndarray, ...]
    demands: tuple[np.ndarray, ...]
    scales: np.ndarray
    capacities: np.ndarray
    platforms: tuple[str, ...]
    dimension_keys: tuple[tuple[str, ...], ...]
    dimension_values: tuple[np.ndarray, ...]
    dimension_weights: tuple[np.ndarray, ...]

    @property
    def n_jobs(self) -> int:
        """Return the number of jobs in the window."""
        return len(self.job_ids)

    @property
    def max_candidates(self) -> int:
        """Return the largest candidate count among the jobs."""
        return max((len(ids) for ids in self.candidate_ids), default=0)

    @property
    def max_dimensions(self) -> int:
        """Return the largest report dimension count among the jobs."""
        return max((len(keys) for keys in self.dimension_keys), default=0)

    def true_values(self, job_index: int) -> np.ndarray:
        """Return the candidate values of one job under its true multipliers."""
        return self.dimension_weights[job_index] @ self.dimension_values[job_index]


def structure_from_window(window: Window) -> WindowStructure:
    """Build the public structure and the true multipliers of a window."""
    platforms = tuple(sorted(window.platforms))
    platform_index = {name: index for index, name in enumerate(platforms)}
    capacities = _read_only(
        np.asarray([window.free_nodes[name] for name in platforms], dtype=np.float32)
    )
    candidate_ids, channels, demands, scales = [], [], [], []
    dimension_keys, dimension_values, dimension_weights = [], [], []
    for job in window.jobs:
        candidates = window.candidates[job.job_id]
        costs = [candidate.cost_credits for candidate in candidates]
        scale = max(min(costs, default=0.0), 1.0e-9)
        scales.append(scale)
        if isinstance(job.bid, Mapping):
            keys = tuple(sorted(job.bid))
            multipliers = np.asarray([job.bid[key] for key in keys], dtype=np.float32)
        else:
            keys = (SINGLE,)
            multipliers = np.asarray([job.bid], dtype=np.float32)
        rows, demand_rows, weight_rows = [], [], []
        for candidate in candidates:
            nodes = demand(job, candidate)
            demand_row = np.zeros(len(platforms), dtype=np.float32)
            for name, count in nodes.items():
                demand_row[platform_index[name]] = float(count)
            demand_rows.append(demand_row)
            shares = [
                window.free_nodes[name] / (window.free_nodes[name] + count)
                for name, count in nodes.items()
            ]
            rows.append(
                (
                    candidate.cost_credits / scale,
                    np.log1p(demand_row.sum()),
                    max(leg.limit_s for leg in job.legs) / 3600.0,
                    min(shares),
                    float(len(job.legs)),
                )
            )
            weights = np.zeros(len(keys), dtype=np.float32)
            if isinstance(job.bid, Mapping):
                for leg, name in zip(job.legs, candidate.platforms):
                    weights[keys.index(name)] += window.platforms[name].cost(
                        leg.num_nodes, leg.limit_s
                    )
            elif job.bid > 0.0:
                weights[0] = candidate.value_credits / job.bid
            weight_rows.append(weights)
        candidate_ids.append(tuple(candidate.id for candidate in candidates))
        channels.append(
            _read_only(
                np.asarray(rows, dtype=np.float32).reshape(len(rows), PUBLIC_CHANNELS)
            )
        )
        demands.append(
            _read_only(
                np.asarray(demand_rows, dtype=np.float32).reshape(
                    len(demand_rows), len(platforms)
                )
            )
        )
        dimension_keys.append(keys)
        dimension_values.append(_read_only(multipliers))
        dimension_weights.append(
            _read_only(
                np.asarray(weight_rows, dtype=np.float32).reshape(
                    len(weight_rows), len(keys)
                )
            )
        )
    return WindowStructure(
        tuple(job.job_id for job in window.jobs),
        tuple(candidate_ids),
        tuple(channels),
        tuple(demands),
        _read_only(np.asarray(scales, dtype=np.float32)),
        capacities,
        platforms,
        tuple(dimension_keys),
        tuple(dimension_values),
        tuple(dimension_weights),
    )


class WindowBatch:
    """Pad window structures into tensors with validity masks."""

    def __init__(
        self,
        structures: Sequence[WindowStructure],
        device: str | torch.device = "cpu",
    ) -> None:
        """Build the tensors of one non-empty batch of windows."""
        if not structures:
            raise ValueError("WindowBatch requires at least one structure")
        self.structures = tuple(structures)
        self.device = torch.device(device)
        self.platforms = tuple(
            sorted({name for structure in structures for name in structure.platforms})
        )
        platform_index = {name: index for index, name in enumerate(self.platforms)}
        b = len(structures)
        n = max(structure.n_jobs for structure in structures)
        s = max(max(structure.max_candidates, 1) for structure in structures)
        d = max(max(structure.max_dimensions, 1) for structure in structures)
        p = len(self.platforms)
        zeros = lambda *shape, **kw: torch.zeros(*shape, device=self.device, **kw)
        self.job_mask = zeros(b, n, dtype=torch.bool)
        self.candidate_mask = zeros(b, n, s, dtype=torch.bool)
        self.dimension_mask = zeros(b, n, d, dtype=torch.bool)
        self.public_channels = zeros(b, n, s, PUBLIC_CHANNELS)
        self.demands = zeros(b, n, s, p)
        self.capacities = zeros(b, p)
        self.scales = zeros(b, n)
        self.dimension_values = zeros(b, n, d)
        self.dimension_weights = zeros(b, n, s, d)
        for i, structure in enumerate(structures):
            local = [platform_index[name] for name in structure.platforms]
            self.capacities[i, local] = torch.tensor(
                structure.capacities, device=self.device
            )
            for j in range(structure.n_jobs):
                c = len(structure.candidate_ids[j])
                k = len(structure.dimension_keys[j])
                self.job_mask[i, j] = True
                self.candidate_mask[i, j, :c] = True
                self.dimension_mask[i, j, :k] = True
                self.public_channels[i, j, :c] = torch.tensor(
                    structure.public_channels[j], device=self.device
                )
                self.demands[i, j, :c, local] = torch.tensor(
                    structure.demands[j], device=self.device
                )
                self.scales[i, j] = float(structure.scales[j])
                self.dimension_values[i, j, :k] = torch.tensor(
                    structure.dimension_values[j], device=self.device
                )
                self.dimension_weights[i, j, :c, :k] = torch.tensor(
                    structure.dimension_weights[j], device=self.device
                )
        self.true_values = self.candidate_values(self.dimension_values)

    @property
    def in_channels(self) -> int:
        """Return the private plus public channel count."""
        return IN_CHANNELS

    @property
    def slot_mask(self) -> torch.Tensor:
        """Return the candidate mask under the network's slot name."""
        return self.candidate_mask

    @property
    def demand(self) -> torch.Tensor:
        """Return the candidate demands under the deployment API's name."""
        return self.demands

    def candidate_values(self, dimension_values: torch.Tensor) -> torch.Tensor:
        """Return every candidate's value under the given multipliers."""
        if dimension_values.shape != self.dimension_values.shape:
            raise ValueError(
                f"dimension_values must have shape {tuple(self.dimension_values.shape)}"
            )
        values = (
            self.dimension_weights.to(dimension_values.dtype)
            * dimension_values.unsqueeze(2)
        ).sum(dim=-1)
        return values * self.candidate_mask.to(values.dtype)

    def values_from_factors(
        self,
        dimension_values: torch.Tensor,
        factors: torch.Tensor,
    ) -> torch.Tensor:
        """Return candidate values after scaling the multipliers by factors."""
        if factors.shape != self.dimension_mask.shape:
            raise ValueError(
                f"factors must have shape {tuple(self.dimension_mask.shape)}"
            )
        return self.candidate_values(dimension_values * factors)

    def input_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """Build the network input: value over scale, then the public channels."""
        if values.shape != self.true_values.shape:
            raise ValueError(f"values must have shape {tuple(self.true_values.shape)}")
        normalized = values / self.scales.unsqueeze(-1).clamp_min(1.0e-9)
        inputs = torch.cat((normalized.unsqueeze(-1), self.public_channels), dim=-1)
        return inputs * self.candidate_mask.unsqueeze(-1).to(inputs.dtype)


def sample_values(
    structures: Sequence[WindowStructure],
    rng: np.random.Generator,
    spec: ValueSamplingSpec,
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Sample multipliers: one lognormal factor per job times a preference per
    report dimension, both seeded."""
    sampled = []
    for structure in structures:
        jobs = []
        for keys in structure.dimension_keys:
            factor = float(rng.lognormal(spec.lognormal_mean, spec.lognormal_sigma))
            jobs.append(
                np.asarray(
                    [
                        factor
                        * float(rng.uniform(spec.preference_low, spec.preference_high))
                        for _ in keys
                    ],
                    dtype=np.float32,
                )
            )
        sampled.append(tuple(jobs))
    return tuple(sampled)


__all__ = [
    "IN_CHANNELS",
    "PUBLIC_CHANNELS",
    "ValueSamplingSpec",
    "WindowBatch",
    "WindowStructure",
    "sample_values",
    "structure_from_window",
]
