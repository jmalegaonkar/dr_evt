################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The mechanism interface: decisions, the abstract mechanism, and candidates."""

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


def candidates(job, platforms, free_nodes) -> dict[str, tuple[float, float]]:
    """Return feasible platform offers as cost and value pairs."""
    offers = {}
    for name, platform in platforms.items():
        price = job.price(name)
        if (
            not platform.fits(job)
            or price is None
            or free_nodes[name] < job.num_nodes
            or price + 1.0e-9 < platform.price_per_node_hour
        ):
            continue
        cost = platform.cost(job)
        value = price * job.num_nodes * job.limit_s / 3600
        offers[name] = (cost, value)
    return offers
