################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""VCG: the placement with the largest net value, priced by displacement."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Decision, Mechanism, Placement, Window, demand

_TIE_BREAK = 1.0e-9


@dataclass(frozen=True)
class _Variable:
    index: int
    job_id: str
    placement: Placement
    nodes: dict[str, int]


def _variables(window: Window, excluded: str | None = None) -> list[_Variable]:
    variables = []
    index = 0
    for job in window.jobs:
        for placement in window.candidates[job.job_id]:
            if job.job_id != excluded and placement.net_credits >= 0.0:
                variables.append(
                    _Variable(index, job.job_id, placement, demand(job, placement))
                )
            index += 1
    return variables


def _solve(
    window: Window,
    time_limit_s: float,
    excluded: str | None = None,
) -> tuple[dict[str, Placement], float]:
    """Return the net-value-maximizing placements and their total net value."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    variables = _variables(window, excluded)
    if not variables:
        return {}, 0.0
    # Minimize the negated net value; the tiny index term makes ties resolve
    # in candidate order, so equal windows always give the same answer.
    objective = np.asarray(
        [
            -variable.placement.net_credits + _TIE_BREAK * variable.index
            for variable in variables
        ]
    )
    rows = []
    upper = []
    for job in window.jobs:
        if job.job_id == excluded:
            continue
        rows.append([1.0 if v.job_id == job.job_id else 0.0 for v in variables])
        upper.append(1.0)
    for name in window.free_nodes:
        rows.append([float(v.nodes.get(name, 0)) for v in variables])
        upper.append(float(window.free_nodes[name]))
    result = milp(
        c=objective,
        integrality=np.ones(len(variables), dtype=int),
        bounds=Bounds(np.zeros(len(variables)), np.ones(len(variables))),
        constraints=LinearConstraint(
            np.asarray(rows), np.zeros(len(rows)), np.asarray(upper)
        ),
        options={"time_limit": time_limit_s, "disp": False, "mip_rel_gap": 0.0},
    )
    if result.x is None or int(result.status) != 0:
        raise RuntimeError(f"VCG did not reach an optimum: {result.message}")
    chosen = {
        v.job_id: v.placement
        for amount, v in zip(result.x, variables)
        if float(amount) >= 0.5
    }
    return chosen, sum(p.net_credits for p in chosen.values())


class Vcg(Mechanism):
    """Maximize net value exactly; charge cost plus the Clarke pivot.

    The window is one MILP: a binary per (job, candidate), at most one per
    job, free nodes per platform, solved to a zero gap. A winner's premium is
    what the other jobs would have gained without it, found by solving the
    window again without that job, so a winner that displaces nobody pays its
    cost only.
    """

    name = "vcg"

    def __init__(self, time_limit_s: float = 10.0) -> None:
        """Bound the wall time of each solve."""
        if isinstance(time_limit_s, bool) or time_limit_s <= 0.0:
            raise ValueError("time_limit_s must be a positive number")
        self.time_limit_s = float(time_limit_s)

    def decide(self, window: Window) -> list[Decision]:
        """Return the welfare-maximizing decisions with pivot charges."""
        chosen, welfare = _solve(window, self.time_limit_s)
        decisions = []
        for job in window.jobs:
            placement = chosen.get(job.job_id)
            if placement is None:
                continue
            _, without = _solve(window, self.time_limit_s, excluded=job.job_id)
            pivot = max(without - (welfare - placement.net_credits), 0.0)
            charge = min(placement.value_credits, placement.cost_credits + pivot)
            decisions.append(Decision(job.job_id, placement, charge))
        return decisions


__all__ = ["Vcg"]
