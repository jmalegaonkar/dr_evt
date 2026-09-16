################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Pure Python types and validation shared by market mechanisms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType


def _require_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_nonnegative_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return number


@dataclass(frozen=True)
class LegSpec:
    """Describe the public resource demand of one job leg."""

    leg_id: str
    num_nodes: int
    limit_s: int

    def __post_init__(self) -> None:
        _require_name(self.leg_id, "leg_id")
        _require_positive_int(self.num_nodes, "num_nodes")
        _require_positive_int(self.limit_s, "limit_s")


@dataclass(frozen=True, init=False)
class LegBid:
    """Store one leg's non-negative values by acceptable platform."""

    leg_id: str
    _values: tuple[tuple[str, float], ...] = field(repr=False)

    def __init__(
        self,
        leg_id: str,
        value_by_platform: Mapping[str, float],
    ) -> None:
        """Copy platform values into a sorted, hashable representation."""
        _require_name(leg_id, "leg_id")
        if not isinstance(value_by_platform, Mapping):
            raise ValueError("value_by_platform must be a mapping")
        values = []
        for platform, value in value_by_platform.items():
            platform = _require_name(platform, "platform name")
            values.append((
                platform,
                _require_nonnegative_float(value, "platform value"),
            ))
        object.__setattr__(self, "leg_id", leg_id)
        object.__setattr__(self, "_values", tuple(sorted(values)))

    @property
    def value_by_platform(self) -> Mapping[str, float]:
        """Return the platform values as a read-only mapping."""
        return MappingProxyType(dict(self._values))


@dataclass(frozen=True)
class JobBid:
    """Collect the private per-platform values for every leg of one job."""

    job_id: str
    legs: tuple[LegBid, ...]

    def __post_init__(self) -> None:
        _require_name(self.job_id, "job_id")
        legs = tuple(self.legs)
        if not legs:
            raise ValueError("a job bid must contain at least one leg")
        if len({leg.leg_id for leg in legs}) != len(legs):
            raise ValueError("job bid leg IDs must be unique")
        object.__setattr__(self, "legs", legs)

    def value_of(self, placement: Placement) -> float | None:
        """Return additive value, or None when a leg rejects its platform."""
        value = 0.0
        platforms = placement.platform_by_leg
        for leg in self.legs:
            platform = platforms.get(leg.leg_id)
            leg_value = leg.value_by_platform.get(platform)
            if leg_value is None:
                return None
            value += leg_value
        return value

    @classmethod
    def single(
        cls,
        job_id: str,
        value_by_platform: Mapping[str, float],
    ) -> JobBid:
        """Construct a one-leg job bid whose leg ID is ``"0"``."""
        return cls(job_id, (LegBid("0", value_by_platform),))


@dataclass(frozen=True, init=False)
class Placement:
    """Describe one complete leg-to-platform assignment and public cost."""

    placement_id: str
    _platforms: tuple[tuple[str, str], ...] = field(repr=False)
    _demands: tuple[tuple[str, int], ...] = field(repr=False)
    resource_cost_credits: float

    def __init__(
        self,
        placement_id: str,
        platform_by_leg: Mapping[str, str],
        demand_by_platform: Mapping[str, int],
        resource_cost_credits: float,
    ) -> None:
        """Validate and copy a placement into hashable representations."""
        _require_name(placement_id, "placement_id")
        if not isinstance(platform_by_leg, Mapping) or not platform_by_leg:
            raise ValueError("platform_by_leg must be a non-empty mapping")
        if not isinstance(demand_by_platform, Mapping):
            raise ValueError("demand_by_platform must be a mapping")

        platforms = []
        for leg_id, platform in platform_by_leg.items():
            platforms.append((
                _require_name(leg_id, "leg_id"),
                _require_name(platform, "platform name"),
            ))
        expected_id = "+".join(platform for _, platform in platforms)
        if placement_id != expected_id:
            raise ValueError(
                f"placement_id must be {expected_id!r} for this leg order"
            )

        demands = []
        for platform, nodes in demand_by_platform.items():
            demands.append((
                _require_name(platform, "platform name"),
                _require_positive_int(nodes, "platform demand"),
            ))
        if {platform for _, platform in platforms} != {
            platform for platform, _ in demands
        }:
            raise ValueError("demand platforms must match assigned platforms")

        object.__setattr__(self, "placement_id", placement_id)
        object.__setattr__(self, "_platforms", tuple(platforms))
        object.__setattr__(self, "_demands", tuple(sorted(demands)))
        object.__setattr__(
            self,
            "resource_cost_credits",
            _require_nonnegative_float(
                resource_cost_credits,
                "resource_cost_credits",
            ),
        )

    @property
    def platform_by_leg(self) -> Mapping[str, str]:
        """Return leg assignments as a read-only mapping in leg order."""
        return MappingProxyType(dict(self._platforms))

    @property
    def demand_by_platform(self) -> Mapping[str, int]:
        """Return aggregate platform demands as a read-only mapping."""
        return MappingProxyType(dict(self._demands))


@dataclass(frozen=True)
class JobOffer:
    """Describe one queued job and its currently feasible placements."""

    job_id: str
    submit_s: int
    legs: tuple[LegSpec, ...]
    candidates: tuple[Placement, ...]

    def __post_init__(self) -> None:
        _require_name(self.job_id, "job_id")
        if not isinstance(self.submit_s, int) or isinstance(self.submit_s, bool):
            raise ValueError("submit_s must be an integer")
        legs = tuple(self.legs)
        candidates = tuple(self.candidates)
        if not legs:
            raise ValueError("a job offer must contain at least one leg")
        leg_ids = tuple(leg.leg_id for leg in legs)
        if len(set(leg_ids)) != len(leg_ids):
            raise ValueError("job offer leg IDs must be unique")
        if len({item.placement_id for item in candidates}) != len(candidates):
            raise ValueError("job offer placement IDs must be unique")
        for candidate in candidates:
            if tuple(candidate.platform_by_leg) != leg_ids:
                raise ValueError("candidate legs must match job legs in order")
            expected_demands: dict[str, int] = {}
            for leg in legs:
                platform = candidate.platform_by_leg[leg.leg_id]
                expected_demands[platform] = (
                    expected_demands.get(platform, 0) + leg.num_nodes
                )
            if dict(candidate.demand_by_platform) != expected_demands:
                raise ValueError("candidate demands must equal assigned leg demands")
        object.__setattr__(self, "legs", legs)
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True)
class MarketObservation:
    """Present one deterministic market window to a mechanism."""

    time_s: int
    window_index: int
    seed: int
    jobs: tuple[JobOffer, ...]
    bids: Mapping[str, JobBid]
    free_nodes: Mapping[str, int]
    truncated_jobs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.time_s, int) or isinstance(self.time_s, bool):
            raise ValueError("time_s must be an integer")
        if (
            not isinstance(self.window_index, int)
            or isinstance(self.window_index, bool)
            or self.window_index < 0
        ):
            raise ValueError("window_index must be a non-negative integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")

        jobs = tuple(self.jobs)
        truncated_jobs = tuple(self.truncated_jobs)
        if len({job.job_id for job in jobs}) != len(jobs):
            raise ValueError("observation job IDs must be unique")
        known_job_ids = {job.job_id for job in jobs}
        if len(set(truncated_jobs)) != len(truncated_jobs):
            raise ValueError("truncated job IDs must be unique")
        if any(job_id not in known_job_ids for job_id in truncated_jobs):
            raise ValueError("truncated job IDs must name observed jobs")
        bids = dict(sorted(self.bids.items()))
        for job in jobs:
            bid = bids.get(job.job_id)
            if bid is None:
                raise ValueError(f"job {job.job_id!r} has no bid")
            if tuple(leg.leg_id for leg in bid.legs) != tuple(
                leg.leg_id for leg in job.legs
            ):
                raise ValueError(f"job {job.job_id!r} bid legs do not match")
            for candidate in job.candidates:
                if bid.value_of(candidate) is None:
                    raise ValueError(
                        f"job {job.job_id!r} has an unacceptable candidate"
                    )

        free_nodes = {}
        for platform, nodes in self.free_nodes.items():
            platform = _require_name(platform, "platform name")
            if (
                not isinstance(nodes, int)
                or isinstance(nodes, bool)
                or nodes < 0
            ):
                raise ValueError("free_nodes values must be non-negative integers")
            free_nodes[platform] = nodes

        object.__setattr__(self, "jobs", jobs)
        object.__setattr__(self, "truncated_jobs", truncated_jobs)
        object.__setattr__(self, "bids", MappingProxyType(bids))
        object.__setattr__(
            self,
            "free_nodes",
            MappingProxyType(dict(sorted(free_nodes.items()))),
        )

    def job(self, job_id: str) -> JobOffer:
        """Return the named offer or raise KeyError."""
        for offer in self.jobs:
            if offer.job_id == job_id:
                return offer
        raise KeyError(f"unknown job {job_id!r}")

    def candidate(self, job_id: str, placement_id: str) -> Placement:
        """Return the named candidate for a job or raise KeyError."""
        for candidate in self.job(job_id).candidates:
            if candidate.placement_id == placement_id:
                return candidate
        raise KeyError(
            f"job {job_id!r} has no placement {placement_id!r}"
        )

    def value(self, job_id: str, placement_id: str) -> float | None:
        """Return the job's value for a candidate placement."""
        candidate = self.candidate(job_id, placement_id)
        return self.bids[job_id].value_of(candidate)

    def net_value(self, job_id: str, placement_id: str) -> float | None:
        """Return private value minus public resource cost for a placement."""
        candidate = self.candidate(job_id, placement_id)
        value = self.bids[job_id].value_of(candidate)
        if value is None:
            return None
        return value - candidate.resource_cost_credits


@dataclass(frozen=True)
class Decision:
    """Select one placement and charge for a job in the current window."""

    job_id: str
    placement_id: str
    charge_credits: float
    score: float | None = None


class Mechanism(ABC):
    """Define a market mechanism over one immutable observation."""

    name: str

    @abstractmethod
    def decide(self, obs: MarketObservation) -> list[Decision]:
        """Return ordered placement and charge decisions for one window."""

    def charge_bounds(
        self,
        obs: MarketObservation,
        job_id: str,
        placement_id: str,
    ) -> tuple[float, float]:
        """Return inclusive resource-cost and private-value charge bounds."""
        candidate = obs.candidate(job_id, placement_id)
        value = obs.value(job_id, placement_id)
        if value is None:
            raise ValueError(
                f"job {job_id!r} does not accept placement {placement_id!r}"
            )
        return candidate.resource_cost_credits, value


@dataclass(frozen=True)
class RejectedDecision:
    """Pair an invalid decision with its stable rejection reason."""

    decision: Decision
    reason: str


def validate_decisions(
    obs: MarketObservation,
    decisions: Sequence[Decision],
) -> tuple[list[Decision], list[RejectedDecision]]:
    """Validate decisions in order without charging rejected capacity.

    Only a job's first decision is considered. A later decision for the same
    job is ``duplicate_job`` even when the first decision was rejected.
    """
    accepted: list[Decision] = []
    rejected: list[RejectedDecision] = []
    seen_jobs: set[str] = set()
    used_nodes = {platform: 0 for platform in obs.free_nodes}

    for decision in decisions:
        reason = None
        if decision.job_id in seen_jobs:
            reason = "duplicate_job"
        else:
            seen_jobs.add(decision.job_id)
            try:
                offer = obs.job(decision.job_id)
            except KeyError:
                reason = "unknown_job"
            else:
                candidate = next(
                    (
                        item
                        for item in offer.candidates
                        if item.placement_id == decision.placement_id
                    ),
                    None,
                )
                if candidate is None:
                    reason = "unknown_placement"
                else:
                    value = obs.bids[decision.job_id].value_of(candidate)
                    charge = float(decision.charge_credits)
                    if value is None or not isfinite(charge) or charge > value:
                        reason = "charge_above_value"
                    elif charge < candidate.resource_cost_credits:
                        reason = "charge_below_cost"
                    elif any(
                        used_nodes.get(platform, 0) + nodes
                        > obs.free_nodes.get(platform, 0)
                        for platform, nodes in candidate.demand_by_platform.items()
                    ):
                        reason = "over_capacity"

        if reason is not None:
            rejected.append(RejectedDecision(decision, reason))
            continue

        accepted.append(decision)
        for platform, nodes in candidate.demand_by_platform.items():
            used_nodes[platform] = used_nodes.get(platform, 0) + nodes

    return accepted, rejected
