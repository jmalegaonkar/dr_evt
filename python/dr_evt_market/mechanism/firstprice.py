################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Pay what you bid: greedy by price under capacity, the winner pays its bid."""

from .base import Decision, Mechanism, offers


class FirstPrice(Mechanism):
    """Greedily allocate net-value offers and charge each winner its value.

    Offers are taken from the highest net value, ties going to the earlier job, then
    to the first platform by name. Once a job is placed, its other offers are skipped.
    """

    name = "firstprice"

    def decide(self, jobs, platforms, free_nodes) -> list[Decision]:
        """Return greedy pay-what-you-bid decisions in batch order."""
        jobs = list(jobs)
        ranked = []
        for job_index, job in enumerate(jobs):
            for name, (cost, value) in offers(job, platforms, free_nodes).items():
                ranked.append((value - cost, job_index, name, value))
        ranked.sort(key=lambda offer: (-offer[0], offer[1], offer[2]))

        remaining = dict(free_nodes)
        assigned = set()
        decisions = {}
        for _, job_index, name, value in ranked:
            job = jobs[job_index]
            if job_index in assigned or remaining[name] < job.num_nodes:
                continue
            assigned.add(job_index)
            remaining[name] -= job.num_nodes
            decisions[job_index] = Decision(job.job_id, name, value)
        return [decisions[index] for index in range(len(jobs)) if index in decisions]
