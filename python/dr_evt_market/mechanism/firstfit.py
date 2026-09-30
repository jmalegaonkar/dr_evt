################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Bid-blind first-fit placement over the federation's exposed slices."""

from .base import Decision, Mechanism, candidates


class FirstFit(Mechanism):
    """Place jobs in arrival order on their cheapest available platform."""

    name = "firstfit"

    def offers(self, job, platforms, free_nodes):
        """Return every candidate at its posted cost, without reading the bid."""
        return {
            name: (platforms[name].cost(job),) * 2
            for name in candidates(job, platforms, free_nodes)
        }

    def decide(self, jobs, platforms, free_nodes) -> list[Decision]:
        """Return bid-blind first-fit decisions in batch order."""
        remaining = dict(free_nodes)
        decisions = []
        for job in jobs:
            offers = self.offers(job, platforms, remaining)
            if not offers:
                continue
            name = min(offers, key=lambda choice: offers[choice][0])
            cost = offers[name][0]
            remaining[name] -= job.num_nodes
            decisions.append(Decision(job.job_id, name, cost))
        return decisions
