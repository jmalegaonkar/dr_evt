################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Convert market observations into padded learned-mechanism tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log
from types import MappingProxyType

import numpy as np
import torch

from ..mechanisms.base import MarketObservation

PUBLIC_CHANNELS = 5
IN_CHANNELS = PUBLIC_CHANNELS + 1
DimensionKey = tuple[str, str, str]


def _read_only(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class ValueSamplingSpec:
    """Configure synthetic job factors and platform preferences."""

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
            raise ValueError(
                "preference_high must be at least preference_low"
            )


@dataclass(frozen=True)
class WindowStructure:
    """Store one observation as ragged, immutable NumPy feature arrays."""

    job_ids: tuple[str, ...]
    candidate_ids: tuple[tuple[str, ...], ...]
    public_channels: tuple[np.ndarray, ...]
    demands: tuple[np.ndarray, ...]
    scales: np.ndarray
    capacities: np.ndarray
    platforms: tuple[str, ...]
    true_values: tuple[np.ndarray, ...]
    dimension_keys: tuple[tuple[DimensionKey, ...], ...]
    dimension_values: tuple[np.ndarray, ...]
    dimension_map: Mapping[DimensionKey, tuple[int, ...]]

    @property
    def n_jobs(self) -> int:
        """Return the number of jobs in the window."""
        return len(self.job_ids)

    @property
    def max_candidates(self) -> int:
        """Return the largest candidate count among the jobs."""
        return max((len(candidates) for candidates in self.candidate_ids), default=0)

    @property
    def max_slots(self) -> int:
        """Return the largest candidate count using network slot terminology."""
        return self.max_candidates

    @property
    def max_dimensions(self) -> int:
        """Return the largest report-dimension count among the jobs."""
        return max((len(keys) for keys in self.dimension_keys), default=0)


def structure_from_observation(obs: MarketObservation) -> WindowStructure:
    """Build the immutable public structure and truthful values of a window."""
    platforms = tuple(sorted(obs.free_nodes))
    platform_index = {name: index for index, name in enumerate(platforms)}
    capacities = _read_only(np.asarray(
        [obs.free_nodes[name] for name in platforms],
        dtype=np.float32,
    ))
    candidate_ids = []
    public_channels = []
    demands = []
    scales = []
    true_values = []
    dimension_keys = []
    dimension_values = []
    dimension_map: dict[DimensionKey, tuple[int, ...]] = {}

    for offer in obs.jobs:
        bid_by_leg = {leg.leg_id: leg for leg in obs.bids[offer.job_id].legs}
        costs = [candidate.resource_cost_credits for candidate in offer.candidates]
        scale = max(min(costs, default=0.0), 1.0e-9)
        scales.append(scale)

        keys = tuple(
            (offer.job_id, leg.leg_id, platform)
            for leg in offer.legs
            for platform in bid_by_leg[leg.leg_id].value_by_platform
        )
        key_index = {key: index for index, key in enumerate(keys)}
        dimension_keys.append(keys)
        dimension_values.append(_read_only(np.asarray(
            [
                bid_by_leg[leg_id].value_by_platform[platform]
                for _, leg_id, platform in keys
            ],
            dtype=np.float32,
        )))

        rows = []
        demand_rows = []
        values = []
        candidates_by_dimension = {key: [] for key in keys}
        for candidate_index, candidate in enumerate(offer.candidates):
            demand = np.zeros(len(platforms), dtype=np.float32)
            for platform, nodes in candidate.demand_by_platform.items():
                demand[platform_index[platform]] = float(nodes)
            demand_rows.append(demand)

            free_shares = [
                obs.free_nodes[platform]
                / (obs.free_nodes[platform] + nodes)
                for platform, nodes in candidate.demand_by_platform.items()
            ]
            rows.append((
                candidate.resource_cost_credits / scale,
                np.log1p(demand.sum()),
                max(leg.limit_s for leg in offer.legs) / 3600.0,
                min(free_shares),
                float(len(offer.legs)),
            ))
            value = obs.bids[offer.job_id].value_of(candidate)
            if value is None:
                raise ValueError(
                    f"job {offer.job_id!r} has an unacceptable candidate"
                )
            values.append(value)
            for leg_id, platform in candidate.platform_by_leg.items():
                key = (offer.job_id, leg_id, platform)
                candidates_by_dimension[key].append(candidate_index)
                if key not in key_index:
                    raise ValueError(f"candidate uses unknown report dimension {key!r}")

        candidate_ids.append(tuple(
            candidate.placement_id for candidate in offer.candidates
        ))
        public_channels.append(_read_only(np.asarray(
            rows,
            dtype=np.float32,
        ).reshape(len(rows), PUBLIC_CHANNELS)))
        demands.append(_read_only(np.asarray(
            demand_rows,
            dtype=np.float32,
        ).reshape(len(demand_rows), len(platforms))))
        true_values.append(_read_only(np.asarray(values, dtype=np.float32)))
        dimension_map.update({
            key: tuple(indices)
            for key, indices in candidates_by_dimension.items()
        })

    return WindowStructure(
        tuple(offer.job_id for offer in obs.jobs),
        tuple(candidate_ids),
        tuple(public_channels),
        tuple(demands),
        _read_only(np.asarray(scales, dtype=np.float32)),
        capacities,
        platforms,
        tuple(true_values),
        tuple(dimension_keys),
        tuple(dimension_values),
        MappingProxyType(dimension_map),
    )


class WindowBatch:
    """Pad window structures into tensors while retaining validity masks."""

    def __init__(
        self,
        structures: Sequence[WindowStructure],
        device: str | torch.device = "cpu",
    ) -> None:
        """Build tensors for one non-empty sequence of window structures."""
        if not structures:
            raise ValueError("WindowBatch requires at least one structure")
        self.structures = tuple(structures)
        self.device = torch.device(device)
        self.platforms = tuple(sorted({
            platform
            for structure in structures
            for platform in structure.platforms
        }))
        platform_index = {name: index for index, name in enumerate(self.platforms)}
        batch_size = len(structures)
        n_jobs = max(structure.n_jobs for structure in structures)
        n_candidates = max(
            max(structure.max_candidates, 1) for structure in structures
        )
        n_dimensions = max(
            max(structure.max_dimensions, 1) for structure in structures
        )

        self.job_mask = torch.zeros(
            batch_size,
            n_jobs,
            dtype=torch.bool,
            device=self.device,
        )
        self.candidate_mask = torch.zeros(
            batch_size,
            n_jobs,
            n_candidates,
            dtype=torch.bool,
            device=self.device,
        )
        self.dimension_mask = torch.zeros(
            batch_size,
            n_jobs,
            n_dimensions,
            dtype=torch.bool,
            device=self.device,
        )
        self.public_channels = torch.zeros(
            batch_size,
            n_jobs,
            n_candidates,
            PUBLIC_CHANNELS,
            device=self.device,
        )
        self.demands = torch.zeros(
            batch_size,
            n_jobs,
            n_candidates,
            len(self.platforms),
            device=self.device,
        )
        self.capacities = torch.zeros(
            batch_size,
            len(self.platforms),
            device=self.device,
        )
        self.scales = torch.zeros(
            batch_size,
            n_jobs,
            device=self.device,
        )
        self.dimension_values = torch.zeros(
            batch_size,
            n_jobs,
            n_dimensions,
            device=self.device,
        )
        self.dimension_use = torch.zeros(
            batch_size,
            n_jobs,
            n_candidates,
            n_dimensions,
            device=self.device,
        )

        for batch_index, structure in enumerate(structures):
            local_platforms = [platform_index[name] for name in structure.platforms]
            self.capacities[batch_index, local_platforms] = torch.tensor(
                structure.capacities,
                dtype=torch.float32,
                device=self.device,
            )
            for job_index in range(structure.n_jobs):
                candidate_count = len(structure.candidate_ids[job_index])
                dimension_count = len(structure.dimension_keys[job_index])
                self.job_mask[batch_index, job_index] = True
                self.candidate_mask[
                    batch_index,
                    job_index,
                    :candidate_count,
                ] = True
                self.dimension_mask[
                    batch_index,
                    job_index,
                    :dimension_count,
                ] = True
                self.public_channels[
                    batch_index,
                    job_index,
                    :candidate_count,
                ] = torch.tensor(
                    structure.public_channels[job_index],
                    dtype=torch.float32,
                    device=self.device,
                )
                local_demand = torch.tensor(
                    structure.demands[job_index],
                    dtype=torch.float32,
                    device=self.device,
                )
                self.demands[
                    batch_index,
                    job_index,
                    :candidate_count,
                    local_platforms,
                ] = local_demand
                self.scales[batch_index, job_index] = float(
                    structure.scales[job_index]
                )
                self.dimension_values[
                    batch_index,
                    job_index,
                    :dimension_count,
                ] = torch.tensor(
                    structure.dimension_values[job_index],
                    dtype=torch.float32,
                    device=self.device,
                )
                for dimension_index, key in enumerate(
                    structure.dimension_keys[job_index]
                ):
                    for candidate_index in structure.dimension_map[key]:
                        self.dimension_use[
                            batch_index,
                            job_index,
                            candidate_index,
                            dimension_index,
                        ] = 1.0
        self.true_values = self.candidate_values(self.dimension_values)

    @property
    def in_channels(self) -> int:
        """Return the private plus public network channel count."""
        return IN_CHANNELS

    @property
    def slot_mask(self) -> torch.Tensor:
        """Return the candidate mask under the network's slot terminology."""
        return self.candidate_mask

    @property
    def demand(self) -> torch.Tensor:
        """Return candidate demand under the deployment API's singular name."""
        return self.demands

    def candidate_values(
        self,
        dimension_values: torch.Tensor,
    ) -> torch.Tensor:
        """Sum the report dimensions used by each candidate placement."""
        if dimension_values.shape != self.dimension_values.shape:
            raise ValueError(
                "dimension_values must have shape "
                f"{tuple(self.dimension_values.shape)}"
            )
        values = (
            self.dimension_use.to(dimension_values.dtype)
            * dimension_values.unsqueeze(2)
        ).sum(dim=-1)
        return values * self.candidate_mask.to(values.dtype)

    def input_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """Build the private-value plus five-public-channel network input."""
        if values.shape != self.true_values.shape:
            raise ValueError(
                f"values must have shape {tuple(self.true_values.shape)}"
            )
        normalized = values / self.scales.unsqueeze(-1).clamp_min(1.0e-9)
        inputs = torch.cat(
            (normalized.unsqueeze(-1), self.public_channels),
            dim=-1,
        )
        return inputs * self.candidate_mask.unsqueeze(-1).to(inputs.dtype)

    def values_from_factors(
        self,
        dimension_values: torch.Tensor,
        factors: torch.Tensor,
    ) -> torch.Tensor:
        """Scale report dimensions and sum their leg values per candidate."""
        if dimension_values.shape != self.dimension_values.shape:
            raise ValueError(
                "dimension_values must have shape "
                f"{tuple(self.dimension_values.shape)}"
            )
        expected = self.dimension_mask.shape
        if factors.shape != expected:
            raise ValueError(f"factors must have shape {tuple(expected)}")
        return self.candidate_values(dimension_values * factors)


def sample_values(
    structures: Sequence[WindowStructure],
    rng: np.random.Generator,
    spec: ValueSamplingSpec,
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Sample one value per report dimension.

    Each job draws one lognormal factor over its cheapest candidate cost. Each
    ``(leg, platform)`` dimension independently draws a uniform preference
    multiplier. The job factor times the preference is divided equally across
    the job's legs.
    """
    sampled_structures = []
    for structure in structures:
        sampled_jobs = []
        for job_index, keys in enumerate(structure.dimension_keys):
            factor = float(rng.lognormal(
                mean=spec.lognormal_mean,
                sigma=spec.lognormal_sigma,
            ))
            leg_count = len({leg_id for _, leg_id, _ in keys})
            dimension_values = np.asarray([
                structure.scales[job_index]
                * factor
                * float(rng.uniform(
                    spec.preference_low,
                    spec.preference_high,
                ))
                / max(leg_count, 1)
                for _ in keys
            ], dtype=np.float32)
            sampled_jobs.append(dimension_values)
        sampled_structures.append(tuple(sampled_jobs))
    return tuple(sampled_structures)
