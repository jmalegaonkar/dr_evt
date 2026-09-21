################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The market's data model: platforms, jobs, placements, windows, decisions.

A platform posts a price per node hour and a set of hardware tags. A job is one
or more legs; each leg needs nodes, a time limit and hardware tags. A job's base
cost is what it would pay on the cheapest platform it can use; its bid is a
multiplier on that base, or one multiplier per platform when the job values
platforms differently. A placement assigns every leg to a platform; its cost is
public, its value comes from the bid. A mechanism sees one window of jobs,
candidates and free nodes, and answers with decisions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string")
    return value


def _positive_int(value: object, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{what} must be a positive integer")
    return value


def _money(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{what} must be a finite non-negative number")
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{what} must be a finite non-negative number")
    return number


@dataclass(frozen=True)
class Platform:
    """One cluster as the market sees it: size, price and hardware."""

    name: str
    total_nodes: int
    price_per_node_hour: float
    hardware: frozenset[str] = frozenset()
    address: str | None = None

    def __post_init__(self) -> None:
        _name(self.name, "platform name")
        _positive_int(self.total_nodes, "total_nodes")
        _money(self.price_per_node_hour, "price_per_node_hour")
        object.__setattr__(self, "hardware", frozenset(self.hardware))

    def cost(self, num_nodes: int, limit_s: int) -> float:
        """Return the posted cost in credits of nodes held for a time limit."""
        return self.price_per_node_hour * num_nodes * limit_s / 3600.0


@dataclass(frozen=True)
class Leg:
    """One part of a job: nodes, time limit and the hardware it needs."""

    leg_id: str
    num_nodes: int
    limit_s: int
    requires: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _name(self.leg_id, "leg_id")
        _positive_int(self.num_nodes, "num_nodes")
        _positive_int(self.limit_s, "limit_s")
        object.__setattr__(self, "requires", frozenset(self.requires))

    def fits(self, platform: Platform) -> bool:
        """Return whether the platform has the hardware and the nodes."""
        return (
            self.requires <= platform.hardware
            and self.num_nodes <= platform.total_nodes
        )


@dataclass(frozen=True)
class Job:
    """One job of the stream: its legs and its bid.

    ``bid`` is a multiplier on the job's base cost, the cost of its cheapest
    usable placement, so ``1.5`` means the job would pay up to one and a half
    times its base cost wherever it runs. A mapping from platform name to
    multiplier values each platform separately: the value of a placement is
    then the sum over legs of that leg's cost on its platform times the
    platform's multiplier, and platforms without a multiplier are not used.
    """

    job_id: str
    submit_s: int
    legs: tuple[Leg, ...]
    bid: float | Mapping[str, float]

    def __post_init__(self) -> None:
        _name(self.job_id, "job_id")
        if not isinstance(self.submit_s, int) or isinstance(self.submit_s, bool):
            raise ValueError("submit_s must be an integer")
        if self.submit_s < 0:
            raise ValueError("submit_s must be non-negative")
        legs = tuple(self.legs)
        if not legs:
            raise ValueError("a job needs at least one leg")
        if len({leg.leg_id for leg in legs}) != len(legs):
            raise ValueError("leg ids must be unique within a job")
        object.__setattr__(self, "legs", legs)
        if isinstance(self.bid, Mapping):
            bid = {
                _name(name, "platform name"): _money(value, "bid")
                for name, value in self.bid.items()
            }
            if not bid:
                raise ValueError("a per-platform bid needs at least one platform")
            object.__setattr__(self, "bid", dict(sorted(bid.items())))
        else:
            object.__setattr__(self, "bid", _money(self.bid, "bid"))

    def multiplier(self, platform: str) -> float | None:
        """Return the multiplier that applies on a platform, or None."""
        if isinstance(self.bid, Mapping):
            return self.bid.get(platform)
        return self.bid


@dataclass(frozen=True)
class Placement:
    """One way to run a job: a platform per leg, its cost and its value."""

    platforms: tuple[str, ...]
    cost_credits: float
    value_credits: float

    def __post_init__(self) -> None:
        platforms = tuple(_name(name, "platform name") for name in self.platforms)
        if not platforms:
            raise ValueError("a placement needs at least one platform")
        object.__setattr__(self, "platforms", platforms)
        object.__setattr__(self, "cost_credits", _money(self.cost_credits, "cost"))
        object.__setattr__(self, "value_credits", _money(self.value_credits, "value"))

    @property
    def id(self) -> str:
        """Return the platforms joined with ``+`` in leg order."""
        return "+".join(self.platforms)

    @property
    def net_credits(self) -> float:
        """Return value minus cost."""
        return self.value_credits - self.cost_credits


@dataclass(frozen=True)
class Window:
    """One clearing window: the platforms, their free nodes, the queue, the
    candidates of every queued job."""

    time_s: int
    index: int
    platforms: Mapping[str, Platform]
    free_nodes: Mapping[str, int]
    jobs: tuple[Job, ...]
    candidates: Mapping[str, tuple[Placement, ...]]

    def __post_init__(self) -> None:
        jobs = tuple(self.jobs)
        if len({job.job_id for job in jobs}) != len(jobs):
            raise ValueError("job ids must be unique within a window")
        object.__setattr__(self, "jobs", jobs)
        object.__setattr__(self, "platforms", dict(sorted(self.platforms.items())))
        if set(self.free_nodes) != set(self.platforms):
            raise ValueError("free_nodes must name exactly the window's platforms")
        object.__setattr__(self, "free_nodes", dict(sorted(self.free_nodes.items())))
        candidates = {
            job.job_id: tuple(self.candidates.get(job.job_id, ())) for job in jobs
        }
        object.__setattr__(self, "candidates", candidates)

    def job(self, job_id: str) -> Job:
        """Return the job with this id or raise KeyError."""
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        raise KeyError(job_id)


@dataclass(frozen=True)
class Decision:
    """Give one job one of its candidate placements at a charge."""

    job_id: str
    placement: Placement
    charge_credits: float


@dataclass(frozen=True)
class Rejection:
    """Record why a job or a decision was not placed."""

    job_id: str
    reason: str
    time_s: int


class Mechanism(ABC):
    """Decide who runs where and what they pay, one window at a time."""

    name: str

    @abstractmethod
    def decide(self, window: Window) -> list[Decision]:
        """Return at most one decision per job, each on one of its candidates."""


def demand(job: Job, placement: Placement) -> dict[str, int]:
    """Return the nodes a placement takes on each platform."""
    nodes: dict[str, int] = {}
    for leg, platform in zip(job.legs, placement.platforms):
        nodes[platform] = nodes.get(platform, 0) + leg.num_nodes
    return nodes


def validate_decisions(
    window: Window,
    decisions: Sequence[Decision],
) -> tuple[list[Decision], list[Rejection]]:
    """Keep the decisions the window can honour, in order, and say why not.

    A job's first decision is the only one considered. A decision is refused
    for ``duplicate_job``, ``unknown_job``, ``unknown_placement``,
    ``charge_above_value``, ``charge_below_cost`` or ``over_capacity``; a
    refused decision consumes no capacity.
    """
    accepted: list[Decision] = []
    rejected: list[Rejection] = []
    seen: set[str] = set()
    used: dict[str, int] = {name: 0 for name in window.free_nodes}

    for decision in decisions:
        reason = None
        job = None
        if decision.job_id in seen:
            reason = "duplicate_job"
        else:
            seen.add(decision.job_id)
            try:
                job = window.job(decision.job_id)
            except KeyError:
                reason = "unknown_job"
        if reason is None and decision.placement not in window.candidates[job.job_id]:
            reason = "unknown_placement"
        if reason is None:
            charge = float(decision.charge_credits)
            placement = decision.placement
            if not isfinite(charge) or charge > placement.value_credits + 1e-9:
                reason = "charge_above_value"
            elif charge < placement.cost_credits - 1e-9:
                reason = "charge_below_cost"
            else:
                needed = demand(job, placement)
                if any(
                    used.get(name, 0) + nodes > window.free_nodes.get(name, 0)
                    for name, nodes in needed.items()
                ):
                    reason = "over_capacity"
        if reason is not None:
            rejected.append(Rejection(decision.job_id, reason, window.time_s))
            continue
        accepted.append(decision)
        for name, nodes in demand(job, decision.placement).items():
            used[name] = used.get(name, 0) + nodes
    return accepted, rejected


__all__ = [
    "Decision",
    "Job",
    "Leg",
    "Mechanism",
    "Placement",
    "Platform",
    "Rejection",
    "Window",
    "demand",
    "validate_decisions",
]
