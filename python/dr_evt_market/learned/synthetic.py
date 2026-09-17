################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Generate deterministic synthetic market windows for learned mechanisms.

Each window has three platforms with uniformly sampled capacities and prices,
two to five jobs, and one or two legs per job. Node demands and limits are
uniform. Job values use the lognormal job factor and independent uniform
``(leg, platform)`` preferences documented by ``sample_values``.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from ..mechanisms.base import (
    JobBid,
    JobOffer,
    LegBid,
    LegSpec,
    MarketObservation,
    Placement,
)
from .windows import (
    ValueSamplingSpec,
    WindowStructure,
    sample_values,
    structure_from_observation,
)


def _offers(
    window_index: int,
    capacities: dict[str, int],
    prices: dict[str, float],
    rng: np.random.Generator,
    n_jobs: int,
) -> tuple[tuple[JobOffer, ...], dict[str, JobBid]]:
    platform_names = tuple(sorted(capacities))
    offers = []
    bids = {}
    for job_index in range(n_jobs):
        job_id = f"w{window_index:04d}-j{job_index:03d}"
        n_legs = 2 if float(rng.random()) < 0.25 else 1
        max_nodes = max(1, min(capacities.values()) // (2 * n_legs))
        legs = tuple(
            LegSpec(
                str(leg_index),
                int(rng.integers(1, max_nodes + 1)),
                int(rng.integers(1, 25)) * 300,
            )
            for leg_index in range(n_legs)
        )
        candidates = []
        for assignment in product(platform_names, repeat=n_legs):
            demand: dict[str, int] = {}
            cost = 0.0
            platforms_by_leg = {}
            for leg, platform in zip(legs, assignment):
                platforms_by_leg[leg.leg_id] = platform
                demand[platform] = demand.get(platform, 0) + leg.num_nodes
                cost += (
                    prices[platform]
                    * leg.num_nodes
                    * leg.limit_s
                    / 3600.0
                )
            if any(nodes > capacities[name] for name, nodes in demand.items()):
                continue
            candidates.append(Placement(
                "+".join(assignment),
                platforms_by_leg,
                demand,
                cost,
            ))
        offers.append(JobOffer(job_id, 0, legs, tuple(candidates)))
        bids[job_id] = JobBid(
            job_id,
            tuple(
                LegBid(leg.leg_id, {name: 1.0 for name in platform_names})
                for leg in legs
            ),
        )
    return tuple(offers), bids


def synthetic_structures(
    count: int = 20,
    *,
    seed: int = 0,
    value_spec: ValueSamplingSpec | None = None,
) -> list[WindowStructure]:
    """Generate seeded platform, job, placement, and value structures."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    spec = value_spec or ValueSamplingSpec()
    structures = []
    for window_index in range(count):
        platform_names = ("alpha", "beta", "gamma")
        capacities = {
            name: int(rng.integers(32, 129))
            for name in platform_names
        }
        prices = {
            name: float(rng.uniform(0.5, 3.0))
            for name in platform_names
        }
        offers, placeholder_bids = _offers(
            window_index,
            capacities,
            prices,
            rng,
            int(rng.integers(2, 6)),
        )
        placeholder = MarketObservation(
            0,
            window_index,
            seed,
            offers,
            placeholder_bids,
            capacities,
        )
        draft = structure_from_observation(placeholder)
        sampled = sample_values((draft,), rng, spec)[0]
        bids = {}
        for job_index, offer in enumerate(offers):
            values = {
                key: float(sampled[job_index][dimension_index])
                for dimension_index, key in enumerate(
                    draft.dimension_keys[job_index]
                )
            }
            bids[offer.job_id] = JobBid(
                offer.job_id,
                tuple(
                    LegBid(
                        leg.leg_id,
                        {
                            platform: values[
                                (offer.job_id, leg.leg_id, platform)
                            ]
                            for platform in platform_names
                        },
                    )
                    for leg in offer.legs
                ),
            )
        observation = MarketObservation(
            0,
            window_index,
            seed,
            offers,
            bids,
            capacities,
        )
        structures.append(structure_from_observation(observation))
    return structures
