################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""RegretFormer as a market mechanism: a learned allocation and learned premiums."""

from .base import Decision, Mechanism, offers


class RegretFormer(Mechanism):
    """Place jobs and set premiums with RegretFormer's network; needs torch."""

    name = "regretformer"

    def __init__(self, checkpoint=None, *, seed=0, **shape) -> None:
        """Load a saved network, or build an untrained one from a seed."""
        from . import learned

        self._learned = learned
        if checkpoint is None:
            self.net = learned.network(seed, **shape)
        else:
            self.net = learned.load(checkpoint)

    def save(self, path, **notes) -> None:
        """Save the network with free-form notes."""
        self._learned.save(self.net, path, **notes)

    def decide(self, jobs, platforms, free_nodes) -> list[Decision]:
        """Round the network's allocation under free nodes and charge premiums."""
        jobs = list(jobs)
        if not jobs:
            return []
        window = self._learned.window([(jobs, free_nodes)], platforms)
        assignment, fractions = self._learned.deploy(self.net, window, window.prices)
        names = list(platforms)
        decisions = []
        for index, job in enumerate(jobs):
            if assignment[0, index] < 0:
                continue
            name = names[assignment[0, index]]
            cost, value = offers(job, platforms, free_nodes)[name]
            charge = cost + fractions[0, index] * (value - cost)
            decisions.append(Decision(job.job_id, name, float(charge)))
        return decisions
