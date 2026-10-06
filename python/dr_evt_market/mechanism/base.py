################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The mechanism interface: decisions, the abstract mechanism, candidates, offers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Two amounts of money closer than this are equal: a bid exactly at the posted price
# is not under it, and a charge may sit on either of its bounds.
TOLERANCE = 1.0e-9


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

    def offers(self, job, platforms, free_nodes):
        """Return the platforms this mechanism may give a job, as cost and top charge."""
        return offers(job, platforms, free_nodes)


def job_value(job, platform) -> float:
    """Return a job's reported value on one platform."""
    price = job.price(platform.name) or 0.0
    speed = platform.job_speed(job) if isinstance(job.bid, dict) else 1.0
    return price * job.num_nodes * job.limit_s / speed / 3600


def candidates(job, platforms, free_nodes) -> list[str]:
    """Return the platforms where a job can be placed now, whatever it bids."""
    return [
        name
        for name, platform in platforms.items()
        if platform.fits(job) and job.num_nodes <= free_nodes[name]
    ]


def offers(job, platforms, free_nodes) -> dict[str, tuple[float, float]]:
    """Return the candidates a job bid on, at any price, as cost and value pairs.

    A bid under the posted price can win too, and then pays in full. A bid of zero or
    less is no bid.
    """
    return {
        name: (platforms[name].cost(job), job_value(job, platforms[name]))
        for name in candidates(job, platforms, free_nodes)
        if (job.price(name) or 0.0) > 0
    }
