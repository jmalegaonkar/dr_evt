################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Seeded synthetic windows for training and tests.

Each window has three platforms with uniform capacities and prices, one of
them with a ``gpu`` tag, and two to five jobs of one or two legs. A fifth of
the legs require ``gpu``. Half of the jobs bid one multiplier, the other half
one multiplier per platform; multipliers come from ``sample_values``.
"""

from __future__ import annotations

import numpy as np

from ..mechanisms import Job, Leg, Platform, build_window
from .windows import ValueSamplingSpec, WindowStructure, structure_from_window


def synthetic_structures(
    count: int = 20,
    *,
    seed: int = 0,
    value_spec: ValueSamplingSpec | None = None,
) -> list[WindowStructure]:
    """Generate seeded windows with sampled multipliers."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    spec = value_spec or ValueSamplingSpec()
    structures = []
    for index in range(count):
        platforms = {
            name: Platform(
                name,
                int(rng.integers(32, 129)),
                float(rng.uniform(0.5, 3.0)),
                frozenset({"cpu", "gpu"} if name == "gamma" else {"cpu"}),
            )
            for name in ("alpha", "beta", "gamma")
        }
        jobs = []
        for job_index in range(int(rng.integers(2, 6))):
            n_legs = 2 if rng.random() < 0.25 else 1
            largest = max(
                1, min(p.total_nodes for p in platforms.values()) // (2 * n_legs)
            )
            legs = tuple(
                Leg(
                    str(leg_index),
                    int(rng.integers(1, largest + 1)),
                    int(rng.integers(1, 25)) * 300,
                    frozenset({"gpu"}) if rng.random() < 0.2 else frozenset(),
                )
                for leg_index in range(n_legs)
            )
            factor = float(rng.lognormal(spec.lognormal_mean, spec.lognormal_sigma))
            if rng.random() < 0.5:
                bid: float | dict[str, float] = factor
            else:
                bid = {
                    name: factor
                    * float(rng.uniform(spec.preference_low, spec.preference_high))
                    for name in platforms
                }
            jobs.append(Job(f"w{index:04d}-j{job_index:03d}", 0, legs, bid))
        free = {name: platform.total_nodes for name, platform in platforms.items()}
        window = build_window(0, index, jobs, platforms, free)
        structures.append(structure_from_window(window))
    return structures


__all__ = ["synthetic_structures"]
