################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The VCG auction, and its affine generalization over public reserves and weights."""

from dataclasses import dataclass

from .base import TOLERANCE, Decision, Mechanism, candidates, offers

# Equally good placements go to the earlier offers: each chosen offer adds this much
# times its position in the batch to the objective.
_TIE_BREAK = 1.0e-9


@dataclass(frozen=True)
class _Variable:
    index: int
    job_index: int
    platform: str
    cost: float
    value: float
    nodes: int
    reserve: float
    weight: float

    @property
    def net(self) -> float:
        return self.weight * (self.value - self.reserve)


def _variables(jobs, platforms, free_nodes, terms, excluded=None):
    variables = []
    index = 0
    for job_index, job in enumerate(jobs):
        for name, (cost, value) in offers(job, platforms, free_nodes).items():
            reserve, weight = terms(job, name, cost)
            # A bid under the reserve would lower the objective: no optimum takes it.
            if job_index != excluded and value + TOLERANCE >= reserve:
                variables.append(
                    _Variable(
                        index,
                        job_index,
                        name,
                        cost,
                        value,
                        job.num_nodes,
                        reserve,
                        weight,
                    )
                )
            index += 1
    return variables


def _solve(jobs, platforms, free_nodes, time_limit_s, terms, excluded=None):
    variables = _variables(jobs, platforms, free_nodes, terms, excluded)
    if not variables:
        return {}, 0.0

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    objective = np.asarray(
        [-variable.net + _TIE_BREAK * variable.index for variable in variables]
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


def _leftover_offer(job, platforms, free_nodes, terms):
    left = offers(job, platforms, free_nodes)
    if not left:
        return None
    name = max(left, key=lambda name: left[name][1] - left[name][0])
    cost, value = left[name]
    reserve, _ = terms(job, name, cost)
    return name, min(reserve, value)


class Vcg(Mechanism):
    """Maximize net value and charge Clarke pivots, then fill the nodes left over."""

    name = "vcg"

    def __init__(self, time_limit_s=10.0) -> None:
        """Set the wall-time limit for each optimization."""
        self.time_limit_s = float(time_limit_s)

    def terms(self, job, platform, cost):
        """Return an offer's reserve and its job's weight: the cost and 1 under VCG."""
        return cost, 1.0

    def decide(self, jobs, platforms, free_nodes) -> list[Decision]:
        """Return welfare-maximizing decisions in batch order."""
        jobs = list(jobs)
        terms = self.terms
        chosen, welfare = _solve(jobs, platforms, free_nodes, self.time_limit_s, terms)
        left = dict(free_nodes)
        for variable in chosen.values():
            left[variable.platform] -= variable.nodes
        decisions = []
        for job_index, job in enumerate(jobs):
            variable = chosen.get(job_index)
            if variable is None:
                # The solve skips a job that still fits only when it adds nothing:
                # it bid the reserve, under it, or a hair above. It displaces
                # nobody, so it pays its reserve (its cost under VCG), or its whole
                # bid if less.
                offer = _leftover_offer(job, platforms, left, terms)
                if offer is not None:
                    left[offer[0]] -= job.num_nodes
                    decisions.append(Decision(job.job_id, *offer))
                continue
            _, without = _solve(
                jobs, platforms, free_nodes, self.time_limit_s, terms, job_index
            )
            pivot = without - (welfare - variable.net)
            if pivot < TOLERANCE:
                pivot = 0.0
            charge = min(variable.value, variable.reserve + pivot / variable.weight)
            decisions.append(Decision(job.job_id, variable.platform, charge))
        return decisions


class Affine(Vcg):
    """VCG over public reserves and weights: an affine maximizer.

    An offer's reserve is its cost marked up, by platform, and a job's weight scales its
    value over the reserve, by size. Both read only public facts, never a bid, so a job
    in the solve cannot move its own terms. A winner pays its reserve plus its Clarke
    pivot over its weight, between the reserve and its value. A platform the window
    contests, whose fitting jobs need more nodes than it has free, takes `markup` or
    its entry in `markups`; one it leaves idle takes `idle` or its entry in
    `idle_markups`, the contested markup by default. On idle nodes a job could bid the
    posted price and still be placed at cost, so a reserve there cannot bind: with
    `idle=0`, only contested platforms are marked up.
    """

    name = "affine"

    def __init__(
        self,
        markup=0.0,
        *,
        markups=None,
        idle=None,
        idle_markups=None,
        weights=None,
        wide=8,
        time_limit_s=10.0,
    ) -> None:
        """Set the markups, contested and idle, and the weights by job size."""
        super().__init__(time_limit_s)
        self.markup = float(markup)
        self.markups = {name: float(m) for name, m in (markups or {}).items()}
        self.idle = None if idle is None else float(idle)
        self.idle_markups = {name: float(m) for name, m in (idle_markups or {}).items()}
        self.weights = {size: float(w) for size, w in (weights or {}).items()}
        self.wide = int(wide)
        self._contested = None
        marked = [self.markup, *self.markups.values(), *self.idle_markups.values()]
        if min(marked + ([] if self.idle is None else [self.idle])) < 0:
            raise ValueError("markups must not be negative")
        if any(weight <= 0 for weight in self.weights.values()):
            raise ValueError("weights must be positive")
        if set(self.weights) - {"small", "wide"}:
            raise ValueError("weights are keyed by 'small' or 'wide'")

    def decide(self, jobs, platforms, free_nodes) -> list[Decision]:
        """Find the contested platforms, then decide as VCG does."""
        jobs = list(jobs)
        demand = dict.fromkeys(platforms, 0)
        for job in jobs:
            for name in candidates(job, platforms, free_nodes):
                demand[name] += job.num_nodes
        self._contested = {name: demand[name] > free_nodes[name] for name in platforms}
        return super().decide(jobs, platforms, free_nodes)

    def terms(self, job, platform, cost):
        """Return the marked-up cost and the weight of the job's size."""
        markup = self.markups.get(platform, self.markup)
        if not self._contested[platform]:
            default = markup if self.idle is None else self.idle
            markup = self.idle_markups.get(platform, default)
        size = "wide" if job.num_nodes >= self.wide else "small"
        return cost * (1.0 + markup), self.weights.get(size, 1.0)
