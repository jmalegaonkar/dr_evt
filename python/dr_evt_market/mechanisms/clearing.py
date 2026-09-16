################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Build market observations and submit accepted placement decisions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import isfinite

from ..platforms.base import (
    InfrastructureFailure,
    PlatformSession,
    PlatformSnapshot,
    SubmitRequest,
)
from .base import (
    Decision,
    JobBid,
    JobOffer,
    LegSpec,
    MarketObservation,
    Placement,
)


@dataclass(frozen=True)
class QueuedJob:
    """Describe one job waiting for a market clearing window."""

    job_id: str
    submit_s: int
    legs: tuple[LegSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("job_id must be a non-empty string")
        if not isinstance(self.submit_s, int) or isinstance(self.submit_s, bool):
            raise ValueError("submit_s must be an integer")
        legs = tuple(self.legs)
        if not legs:
            raise ValueError("a queued job must contain at least one leg")
        if len({leg.leg_id for leg in legs}) != len(legs):
            raise ValueError("queued job leg IDs must be unique")
        object.__setattr__(self, "legs", legs)


def _platform_state(
    time_s: int,
    snapshots: Mapping[str, PlatformSnapshot],
    prices: Mapping[str, float],
) -> tuple[tuple[str, ...], dict[str, int], dict[str, float]]:
    platform_names = tuple(sorted(snapshots))
    free_nodes: dict[str, int] = {}
    normalized_prices: dict[str, float] = {}
    for name in platform_names:
        snapshot = snapshots[name]
        if snapshot.name != name:
            raise ValueError(f"snapshot {name!r} reports name {snapshot.name!r}")
        if snapshot.time_s != time_s:
            raise ValueError(
                f"snapshot {name!r} is at {snapshot.time_s}, not {time_s}"
            )
        if snapshot.free_nodes < 0:
            raise ValueError(f"snapshot {name!r} has negative free nodes")
        try:
            price = float(prices[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"platform {name!r} has no valid price") from error
        if not isfinite(price) or price < 0.0:
            raise ValueError(f"platform {name!r} price must be finite and non-negative")
        free_nodes[name] = snapshot.free_nodes
        normalized_prices[name] = price
    return platform_names, free_nodes, normalized_prices


def _candidate(
    job: QueuedJob,
    assignment: tuple[str, ...],
    free_nodes: Mapping[str, int],
    prices: Mapping[str, float],
) -> Placement | None:
    platform_by_leg = {
        leg.leg_id: platform
        for leg, platform in zip(job.legs, assignment)
    }
    demand_by_platform: dict[str, int] = {}
    resource_cost = 0.0
    for leg, platform in zip(job.legs, assignment):
        demand_by_platform[platform] = (
            demand_by_platform.get(platform, 0) + leg.num_nodes
        )
        resource_cost += (
            prices[platform] * leg.num_nodes * leg.limit_s / 3600.0
        )
    if any(
        nodes > free_nodes[platform]
        for platform, nodes in demand_by_platform.items()
    ):
        return None
    return Placement(
        "+".join(assignment),
        platform_by_leg,
        demand_by_platform,
        resource_cost,
    )


def build_observation(
    time_s: int,
    window_index: int,
    seed: int,
    queue: Sequence[QueuedJob],
    bids: Mapping[str, JobBid],
    snapshots: Mapping[str, PlatformSnapshot],
    prices: Mapping[str, float],
    *,
    max_candidates: int = 64,
) -> MarketObservation:
    """Build feasible placements from live free-node snapshots."""
    if (
        not isinstance(max_candidates, int)
        or isinstance(max_candidates, bool)
        or max_candidates < 1
    ):
        raise ValueError("max_candidates must be a positive integer")
    platform_names, free_nodes, normalized_prices = _platform_state(
        time_s,
        snapshots,
        prices,
    )
    offers = []
    truncated_jobs = []
    for job in queue:
        try:
            bid = bids[job.job_id]
        except KeyError as error:
            raise ValueError(f"job {job.job_id!r} has no bid") from error
        bid_by_leg = {leg.leg_id: leg for leg in bid.legs}
        choices = []
        for leg in job.legs:
            leg_bid = bid_by_leg.get(leg.leg_id)
            if leg_bid is None:
                raise ValueError(f"job {job.job_id!r} bid legs do not match")
            choices.append(tuple(
                platform
                for platform in platform_names
                if platform in leg_bid.value_by_platform
                and leg.num_nodes <= free_nodes[platform]
            ))

        candidates = []
        was_truncated = False
        for assignment in product(*choices):
            candidate = _candidate(
                job,
                assignment,
                free_nodes,
                normalized_prices,
            )
            if candidate is None:
                continue
            if len(candidates) == max_candidates:
                was_truncated = True
                break
            candidates.append(candidate)
        if was_truncated:
            truncated_jobs.append(job.job_id)
        offers.append(JobOffer(
            job.job_id,
            job.submit_s,
            job.legs,
            tuple(candidates),
        ))

    return MarketObservation(
        time_s,
        window_index,
        seed,
        tuple(offers),
        bids,
        free_nodes,
        tuple(truncated_jobs),
    )


def submit_decisions(
    decisions: Sequence[Decision],
    obs: MarketObservation,
    platforms: Mapping[str, PlatformSession],
    time_s: int,
) -> dict[tuple[str, str], int]:
    """Submit accepted decisions once per platform and return leg handles."""
    batches: dict[str, list[SubmitRequest]] = {
        name: [] for name in sorted(platforms)
    }
    batch_keys: dict[str, list[tuple[str, str]]] = {
        name: [] for name in sorted(platforms)
    }
    ordered_keys = []
    for decision in decisions:
        offer = obs.job(decision.job_id)
        candidate = obs.candidate(decision.job_id, decision.placement_id)
        for leg in offer.legs:
            platform = candidate.platform_by_leg[leg.leg_id]
            if platform not in batches:
                raise ValueError(f"unknown platform {platform!r}")
            key = (decision.job_id, leg.leg_id)
            ordered_keys.append(key)
            batches[platform].append(SubmitRequest(
                key=f"{decision.job_id}/{leg.leg_id}",
                submit_s=time_s,
                num_nodes=leg.num_nodes,
                limit_s=leg.limit_s,
            ))
            batch_keys[platform].append(key)

    handles_by_leg = {}
    for name in sorted(platforms):
        handles = platforms[name].submit(batches[name])
        if len(handles) != len(batch_keys[name]):
            raise InfrastructureFailure(
                f"platform {name!r} returned {len(handles)} handles for "
                f"{len(batch_keys[name])} submitted legs"
            )
        handles_by_leg.update(zip(batch_keys[name], handles))
    return {key: handles_by_leg[key] for key in ordered_keys}
