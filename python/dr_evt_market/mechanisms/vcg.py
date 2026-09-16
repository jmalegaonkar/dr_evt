################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""VCG allocation by exact net welfare with a bounded greedy fallback."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from .base import Decision, MarketObservation, Mechanism, Placement

_MILP_VARIABLE_BUDGET = 800
_TIE_BREAK_WEIGHT = 1.0e-9
_BISECTION_STEPS = 60


@dataclass(frozen=True)
class _Variable:
    index: int
    job_id: str
    candidate: Placement
    value: float
    net_value: float


def _eligible_variables(
    obs: MarketObservation,
    excluded_job: str | None = None,
) -> list[_Variable]:
    variables = []
    index = 0
    for offer in obs.jobs:
        for candidate in sorted(
            offer.candidates,
            key=lambda item: item.placement_id,
        ):
            value = obs.bids[offer.job_id].value_of(candidate)
            if value is not None:
                net_value = value - candidate.resource_cost_credits
                if net_value >= 0.0 and offer.job_id != excluded_job:
                    variables.append(_Variable(
                        index,
                        offer.job_id,
                        candidate,
                        value,
                        net_value,
                    ))
            index += 1
    return variables


def _fits(
    variable: _Variable,
    remaining: Mapping[str, int],
) -> bool:
    return all(
        nodes <= remaining.get(platform, 0)
        for platform, nodes in variable.candidate.demand_by_platform.items()
    )


def _solve_exact(
    obs: MarketObservation,
    time_limit_s: float,
    excluded_job: str | None = None,
) -> tuple[dict[str, _Variable], float]:
    from scipy.optimize import Bounds, LinearConstraint, milp
    import numpy as np

    variables = _eligible_variables(obs, excluded_job)
    if not variables:
        return {}, 0.0

    objective = np.asarray([
        -variable.net_value + _TIE_BREAK_WEIGHT * variable.index
        for variable in variables
    ])
    rows = []
    upper = []
    for offer in obs.jobs:
        if offer.job_id == excluded_job:
            continue
        rows.append(np.asarray([
            1.0 if variable.job_id == offer.job_id else 0.0
            for variable in variables
        ]))
        upper.append(1.0)
    for platform in sorted(obs.free_nodes):
        rows.append(np.asarray([
            float(variable.candidate.demand_by_platform.get(platform, 0))
            for variable in variables
        ]))
        upper.append(float(obs.free_nodes[platform]))

    matrix = np.vstack(rows)
    result = milp(
        c=objective,
        integrality=np.ones(len(variables), dtype=int),
        bounds=Bounds(
            np.zeros(len(variables)),
            np.ones(len(variables)),
        ),
        constraints=LinearConstraint(
            matrix,
            np.zeros(len(rows)),
            np.asarray(upper),
        ),
        options={
            "time_limit": time_limit_s,
            "mip_rel_gap": 0.0,
            "disp": False,
        },
    )
    if result.x is None or int(result.status) != 0:
        raise RuntimeError(f"VCG allocation did not reach an optimum: {result.message}")
    selected = {
        variable.job_id: variable
        for amount, variable in zip(result.x, variables)
        if float(amount) >= 0.5
    }
    welfare = sum(variable.net_value for variable in selected.values())
    return selected, welfare


def _greedy_allocation(
    variables: list[_Variable],
    free_nodes: Mapping[str, int],
    overrides: Mapping[int, float] | None = None,
) -> dict[str, _Variable]:
    effective = overrides or {}
    ordered = sorted(
        variables,
        key=lambda variable: (
            -effective.get(variable.index, variable.net_value)
            / sum(variable.candidate.demand_by_platform.values()),
            variable.index,
        ),
    )
    remaining = dict(free_nodes)
    selected = {}
    for variable in ordered:
        if variable.job_id in selected or not _fits(variable, remaining):
            continue
        selected[variable.job_id] = variable
        for platform, nodes in variable.candidate.demand_by_platform.items():
            remaining[platform] -= nodes
    return selected


def _greedy_critical_value(
    winner: _Variable,
    variables: list[_Variable],
    free_nodes: Mapping[str, int],
) -> float:
    def wins_at(net_value: float) -> bool:
        selected = _greedy_allocation(
            variables,
            free_nodes,
            {winner.index: net_value},
        )
        selected_variable = selected.get(winner.job_id)
        return (
            selected_variable is not None
            and selected_variable.index == winner.index
        )

    if wins_at(0.0):
        return 0.0
    low = 0.0
    high = winner.net_value
    for _ in range(_BISECTION_STEPS):
        midpoint = (low + high) / 2.0
        if wins_at(midpoint):
            high = midpoint
        else:
            low = midpoint
    return high


class Vcg(Mechanism):
    """Allocate net welfare and charge Clarke pivot payments.

    Exact windows use a MILP solved to a zero relative gap. The objective
    subtracts ``1e-9 * index`` from each candidate's net value so equal values
    follow placement ID order. Windows above 800 eligible variables use
    deterministic greedy density and bisection-based critical-value payments.
    """

    name = "vcg"

    def __init__(self, time_limit_s: float = 10.0) -> None:
        """Set the positive wall-time limit for each exact MILP solve."""
        if (
            isinstance(time_limit_s, bool)
            or not isinstance(time_limit_s, Real)
            or not isfinite(float(time_limit_s))
            or float(time_limit_s) <= 0.0
        ):
            raise ValueError("time_limit_s must be a positive finite number")
        self.time_limit_s = float(time_limit_s)

    def decide(self, obs: MarketObservation) -> list[Decision]:
        """Return feasible allocations and individually rational charges."""
        variables = _eligible_variables(obs)
        if len(variables) > _MILP_VARIABLE_BUDGET:
            return self._decide_greedy(obs, variables)

        selected, welfare = _solve_exact(obs, self.time_limit_s)
        decisions = []
        for offer in obs.jobs:
            winner = selected.get(offer.job_id)
            if winner is None:
                continue
            _, welfare_without = _solve_exact(
                obs,
                self.time_limit_s,
                excluded_job=offer.job_id,
            )
            others_with_winner = welfare - winner.net_value
            pivot = max(welfare_without - others_with_winner, 0.0)
            charge = min(
                winner.value,
                winner.candidate.resource_cost_credits + pivot,
            )
            decisions.append(Decision(
                offer.job_id,
                winner.candidate.placement_id,
                charge,
                welfare,
            ))
        return decisions

    def _decide_greedy(
        self,
        obs: MarketObservation,
        variables: list[_Variable],
    ) -> list[Decision]:
        selected = _greedy_allocation(variables, obs.free_nodes)
        welfare = sum(variable.net_value for variable in selected.values())
        decisions = []
        for offer in obs.jobs:
            winner = selected.get(offer.job_id)
            if winner is None:
                continue
            critical_value = _greedy_critical_value(
                winner,
                variables,
                obs.free_nodes,
            )
            charge = min(
                winner.value,
                winner.candidate.resource_cost_credits + critical_value,
            )
            decisions.append(Decision(
                offer.job_id,
                winner.candidate.placement_id,
                charge,
                welfare,
            ))
        return decisions
