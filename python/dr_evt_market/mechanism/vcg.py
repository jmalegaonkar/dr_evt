################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The VCG auction: exact net-value maximization with Clarke pivot charges."""

from dataclasses import dataclass

from .base import Decision, Mechanism, candidates

_TIE_BREAK = 1.0e-9


@dataclass(frozen=True)
class _Variable:
    index: int
    job_index: int
    platform: str
    cost: float
    value: float
    nodes: int

    @property
    def net(self) -> float:
        return self.value - self.cost


def _variables(jobs, platforms, free_nodes, excluded=None):
    variables = []
    index = 0
    for job_index, job in enumerate(jobs):
        for name, (cost, value) in candidates(job, platforms, free_nodes).items():
            if job_index != excluded:
                variables.append(
                    _Variable(index, job_index, name, cost, value, job.num_nodes)
                )
            index += 1
    return variables


def _solve(jobs, platforms, free_nodes, time_limit_s, excluded=None):
    variables = _variables(jobs, platforms, free_nodes, excluded)
    if not variables:
        return {}, 0.0

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    objective = np.asarray(
        [
            -variable.net + _TIE_BREAK * variable.index - _TIE_BREAK
            for variable in variables
        ]
    )
    rows = []
    upper = []
    for job_index in range(len(jobs)):
        rows.append([float(v.job_index == job_index) for v in variables])
        upper.append(1.0)
    for name in platforms:
        rows.append([float(v.nodes) if v.platform == name else 0.0 for v in variables])
        upper.append(float(free_nodes[name]))
    result = milp(
        c=objective,
        integrality=np.ones(len(variables), dtype=int),
        bounds=Bounds(np.zeros(len(variables)), np.ones(len(variables))),
        constraints=LinearConstraint(
            np.asarray(rows), np.zeros(len(rows)), np.asarray(upper)
        ),
        options={"mip_rel_gap": 0.0, "time_limit": time_limit_s, "disp": False},
    )
    if result.x is None or int(result.status) != 0:
        raise RuntimeError(f"VCG did not reach an optimum: {result.message}")
    chosen = {
        variable.job_index: variable
        for amount, variable in zip(result.x, variables)
        if float(amount) >= 0.5
    }
    return chosen, sum(variable.net for variable in chosen.values())


class Vcg(Mechanism):
    """Maximize net value and charge each winner its Clarke pivot."""

    name = "vcg"

    def __init__(self, time_limit_s=10.0) -> None:
        """Set the wall-time limit for each optimization."""
        self.time_limit_s = float(time_limit_s)

    def decide(self, jobs, platforms, free_nodes) -> list[Decision]:
        """Return welfare-maximizing decisions in batch order."""
        jobs = list(jobs)
        chosen, welfare = _solve(jobs, platforms, free_nodes, self.time_limit_s)
        decisions = []
        for job_index, job in enumerate(jobs):
            variable = chosen.get(job_index)
            if variable is None:
                continue
            _, without = _solve(
                jobs, platforms, free_nodes, self.time_limit_s, job_index
            )
            pivot = max(without - (welfare - variable.net), 0.0)
            charge = min(variable.value, variable.cost + pivot)
            decisions.append(Decision(job.job_id, variable.platform, charge))
        return decisions
