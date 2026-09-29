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


def job_value(job, platform) -> float:
    """Return a job's reported value on one platform."""
    price = job.price(platform.name) or 0.0
    seconds = job.requested_s or job.limit_s
    speed = platform.job_speed(job) if isinstance(job.bid, dict) else 1.0
    return price * job.num_nodes * seconds / speed / 3600


def candidates(job, platforms, free_nodes) -> dict[str, tuple[float, float]]:
    """Return feasible platform offers as cost and value pairs."""
    offers = {}
    for name, platform in platforms.items():
        price = job.price(name)
        if not platform.fits(job) or price is None or free_nodes[name] < job.num_nodes:
            continue
        cost = platform.cost(job)
        value = job_value(job, platform)
        if value + 1.0e-9 < cost:
            continue
        offers[name] = (cost, value)
    return offers
