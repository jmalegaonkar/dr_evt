################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The mechanism interface: decisions, the abstract mechanism, base cost, candidates."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """One winning job, its platform and its charge."""

    job_id: str
    platform: str
    charge: float


class Mechanism(ABC):
    """Choose platform assignments and charges for a batch of jobs."""

    name: str

    @abstractmethod
    def decide(self, jobs, platforms, free_nodes) -> list[Decision]:
        """Return the winning decisions in batch order."""


def base_cost(job, platforms) -> float | None:
    """Return the cheapest cost among platforms that can fit a job."""
    costs = (
        platform.cost(job) for platform in platforms.values() if platform.fits(job)
    )
    return min(costs, default=None)


def candidates(job, platforms, free_nodes) -> dict[str, tuple[float, float]]:
    """Return feasible platform offers as cost and value pairs."""
    cheapest = base_cost(job, platforms)
    offers = {}
    for name, platform in platforms.items():
        multiplier = job.multiplier(name)
        if (
            not platform.fits(job)
            or multiplier is None
            or free_nodes[name] < job.num_nodes
        ):
            continue
        cost = platform.cost(job)
        value = (
            multiplier * cost if isinstance(job.bid, dict) else multiplier * cheapest
        )
        if value + 1.0e-9 >= cost:
            offers[name] = (cost, value)
    return offers
