################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""From the queue and the platforms to a window, and from decisions to jobs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

from ..platforms.base import InfrastructureFailure, PlatformSession, SubmitRequest
from .base import Decision, Job, Placement, Platform, Window


def placements(
    job: Job,
    platforms: Mapping[str, Platform],
    free_nodes: Mapping[str, int] | None = None,
) -> tuple[Placement, ...]:
    """Return every placement the job could take, with cost and value.

    A leg may go to a platform that has its hardware, enough nodes overall and,
    when ``free_nodes`` is given, enough free nodes now; a job with a
    per-platform bid may only use the platforms it bid on. Legs sharing a
    platform must fit together. Platforms are tried in name order, so the
    result is deterministic. With ``free_nodes`` given, placements whose value
    does not cover their cost are left out, since no mechanism may take them.
    """
    names = sorted(platforms)
    choices = []
    for leg in job.legs:
        usable = [
            name
            for name in names
            if leg.fits(platforms[name])
            and job.multiplier(name) is not None
            and (free_nodes is None or leg.num_nodes <= free_nodes.get(name, 0))
        ]
        if not usable:
            return ()
        choices.append(usable)

    single = not isinstance(job.bid, Mapping)
    base = base_cost(job, platforms) if single else None
    found = []
    for assignment in product(*choices):
        nodes: dict[str, int] = {}
        cost = 0.0
        value = 0.0
        for leg, name in zip(job.legs, assignment):
            nodes[name] = nodes.get(name, 0) + leg.num_nodes
            leg_cost = platforms[name].cost(leg.num_nodes, leg.limit_s)
            cost += leg_cost
            if not single:
                value += job.multiplier(name) * leg_cost
        if any(
            count > platforms[name].total_nodes
            or (free_nodes is not None and count > free_nodes.get(name, 0))
            for name, count in nodes.items()
        ):
            continue
        if single:
            value = job.bid * base
        if free_nodes is not None and value < cost:
            continue
        found.append(Placement(tuple(assignment), cost, value))
    return tuple(found)


def base_cost(job: Job, platforms: Mapping[str, Platform]) -> float | None:
    """Return the cost of the job's cheapest usable placement, or None."""
    names = sorted(platforms)
    cheapest = None
    choices = []
    for leg in job.legs:
        usable = [
            name
            for name in names
            if leg.fits(platforms[name]) and job.multiplier(name) is not None
        ]
        if not usable:
            return None
        choices.append(usable)
    for assignment in product(*choices):
        nodes: dict[str, int] = {}
        cost = 0.0
        for leg, name in zip(job.legs, assignment):
            nodes[name] = nodes.get(name, 0) + leg.num_nodes
            cost += platforms[name].cost(leg.num_nodes, leg.limit_s)
        if any(count > platforms[name].total_nodes for name, count in nodes.items()):
            continue
        if cheapest is None or cost < cheapest:
            cheapest = cost
    return cheapest


def build_window(
    time_s: int,
    index: int,
    queue: Sequence[Job],
    platforms: Mapping[str, Platform],
    free_nodes: Mapping[str, int],
) -> Window:
    """Offer every queued job its placements in the free nodes of this window."""
    return Window(
        time_s,
        index,
        dict(platforms),
        dict(free_nodes),
        tuple(queue),
        {job.job_id: placements(job, platforms, free_nodes) for job in queue},
    )


def submit(
    decisions: Sequence[Decision],
    window: Window,
    sessions: Mapping[str, PlatformSession],
    time_s: int,
) -> dict[tuple[str, str], int]:
    """Send every accepted leg to its platform, one batch per platform.

    Returns the platform handle of every leg, keyed by job id and leg id.
    """
    names = sorted(sessions)
    batches: dict[str, list[SubmitRequest]] = {name: [] for name in names}
    keys: dict[str, list[tuple[str, str]]] = {name: [] for name in names}
    for decision in decisions:
        job = window.job(decision.job_id)
        for leg, name in zip(job.legs, decision.placement.platforms):
            if name not in batches:
                raise ValueError(f"unknown platform {name!r}")
            batches[name].append(
                SubmitRequest(
                    key=f"{job.job_id}/{leg.leg_id}",
                    submit_s=time_s,
                    num_nodes=leg.num_nodes,
                    limit_s=leg.limit_s,
                )
            )
            keys[name].append((job.job_id, leg.leg_id))
    handles: dict[tuple[str, str], int] = {}
    for name in names:
        if not batches[name]:
            continue
        returned = sessions[name].submit(batches[name])
        if len(returned) != len(keys[name]):
            raise InfrastructureFailure(
                f"platform {name!r} returned {len(returned)} handles "
                f"for {len(keys[name])} legs"
            )
        handles.update(zip(keys[name], returned))
    return handles


__all__ = ["base_cost", "build_window", "placements", "submit"]
