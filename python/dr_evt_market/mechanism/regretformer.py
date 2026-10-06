################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""RegretFormer as a market mechanism: a learned allocation and learned premiums."""

from .base import Decision, Mechanism


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
        """Round the network's allocation under free nodes and charge each winner."""
        jobs = list(jobs)
        if not jobs:
            return []
        window = self._learned.window([(jobs, free_nodes)], platforms)
        assignment, charges = self._learned.deploy(self.net, window, window.prices)
        names = list(platforms)
        return [
            Decision(job.job_id, names[column], float(charges[0, index, column]))
            for index, (job, column) in enumerate(zip(jobs, assignment[0]))
            if column >= 0
        ]
