################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The window loop: prefix of the queue, auction, winners to their platforms."""

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from .mechanism import Decision, Mechanism, candidates
from .mechanism.base import TOLERANCE, job_value

# A queued job's credit grows once for every whole period it has waited.
_AGING_PERIOD_S = 600


@dataclass(frozen=True)
class RoutedJob:
    """One winning job, its route, its charge and the federation's part of it."""

    job_id: str
    platform: str
    window: int
    cost: float
    value: float
    premium: float
    charge: float
    subsidy: float
    submit_s: int
    begin_s: int
    end_s: int


@dataclass(frozen=True)
class Waiting:
    """One job still waiting when the run ends, and what keeps it from running."""

    job_id: str
    reason: str
    submit_s: int


@dataclass(frozen=True)
class _Service:
    """Service measures for one community or all routed jobs."""

    community: str
    count: int
    mean_wait_s: float
    node_hour_weighted_mean_wait_s: float
    mean_bounded_slowdown: float


@dataclass
class Result:
    """The routes, the jobs still waiting, statistics and resolved configuration."""

    routed: list[RoutedJob]
    waiting: list[Waiting]
    statistics: dict[str, dict[str, float]]
    configuration: dict
    communities: dict[str, tuple[str, int]]


class MarketError(RuntimeError):
    """A mechanism or platform violated the market's placement guarantee."""


def _check_decisions(
    batch, platforms, free_nodes, mechanism, decisions: list[Decision]
):
    batch_by_id = {job.job_id: job for job in batch}
    seen = set()
    used = {name: 0 for name in platforms}
    checked = {}
    for decision in decisions:
        if decision.job_id in seen:
            raise MarketError(f"{decision.job_id}: decided twice")
        seen.add(decision.job_id)
        job = batch_by_id.get(decision.job_id)
        if job is None:
            raise MarketError(f"{decision.job_id}: not in the batch")
        offer = mechanism.offers(job, platforms, free_nodes).get(decision.platform)
        if offer is None:
            raise MarketError(f"{decision.job_id}: no offer on this platform")
        cost, maximum_charge = offer
        value = job_value(job, platforms[decision.platform])
        # A winner pays at least the cost, or its whole bid when that is less.
        least = min(cost, maximum_charge)
        if not least - TOLERANCE <= decision.charge <= maximum_charge + TOLERANCE:
            raise MarketError(f"{decision.job_id}: charge is outside its offer")
        used[decision.platform] += job.num_nodes
        if used[decision.platform] > free_nodes[decision.platform]:
            raise MarketError(f"{decision.job_id}: platform capacity exceeded")
        checked[decision.job_id] = (job, decision, cost, value)
    left = {name: free_nodes[name] - used[name] for name in platforms}
    for job in batch:
        if job.job_id not in checked and mechanism.offers(job, platforms, left):
            raise MarketError(f"{job.job_id}: left waiting on free nodes")
    return [checked[job.job_id] for job in batch if job.job_id in checked]


def _auction(able, platforms, free_nodes, mechanism, prefix, max_prefix):
    """Auction the first jobs that can be placed, then refill the nodes left free.

    The first auction takes the first `prefix` jobs. While the auctions leave free
    nodes that a later job, among the first `max_prefix`, bid on, the next `prefix`
    jobs that fit those nodes are auctioned for them. Earlier winners keep theirs.
    """
    winners = []
    left = dict(free_nodes)
    batch, later = able[:prefix], able[prefix:max_prefix]
    while True:
        decisions = list(mechanism.decide(batch, platforms, dict(left)))
        placed = _check_decisions(batch, platforms, left, mechanism, decisions)
        winners += placed
        for job, decision, _, _ in placed:
            left[decision.platform] -= job.num_nodes
        if not any(mechanism.offers(job, platforms, left) for job in later):
            return winners
        batch = [job for job in later if candidates(job, platforms, left)][:prefix]
        taken = {job.job_id for job in batch}
        later = [job for job in later if job.job_id not in taken]


def _credit(job, t, aging, aging_cap) -> float:
    """Return a queued job's credit at time t, in the posted prices' unit."""
    return min(aging * ((t - job.submit_s) // _AGING_PERIOD_S), aging_cap)


def _offered(job, platforms, credit, under_price):
    """Return the job as an auction sees it: its credit added to its value on each
    platform it bid on, and without `under_price`, no bid under a platform's price."""
    if credit <= 0 and under_price:
        return job
    bid = {}
    for name, platform in platforms.items():
        price = job.price(name) or 0.0
        if price > 0 and platform.fits(job):
            # Prices are per node-hour of the platform, where a single bid's price
            # scales by speed, and the credit spreads over the node-hours used there.
            speed = platform.job_speed(job)
            node_hours = job.num_nodes * job.limit_s / speed / 3600
            scale = 1.0 if isinstance(job.bid, dict) else speed
            offered = price * scale + credit / node_hours
            if under_price or offered * node_hours + TOLERANCE >= platform.cost(job):
                bid[name] = offered
    return replace(job, bid=bid)


def _configuration(
    platforms, mechanism, window_s, prefix, max_prefix, aging, cap, under_price
):
    return {
        "mechanism": mechanism.name,
        "window_s": window_s,
        "prefix": prefix,
        "max_prefix": max_prefix,
        "aging": aging,
        "aging_cap": cap,
        "under_price": under_price,
        "platforms": {
            name: {
                "total_nodes": platform.total_nodes,
                "exposed_nodes": platform.exposed_nodes,
                "price_per_node_hour": platform.price_per_node_hour,
                "hardware": sorted(platform.hardware),
                "speed": platform.speed,
            }
            for name, platform in platforms.items()
        },
    }


def _blocked_by(job, platforms, full_nodes, mechanism):
    if not any(job.requires <= platform.hardware for platform in platforms.values()):
        return "hardware"
    if not any(platform.fits(job) for platform in platforms.values()):
        return "oversize"
    if not mechanism.offers(job, platforms, full_nodes):
        return "no_bid"
    return None


def run(
    jobs,
    platforms,
    mechanism: Mechanism,
    *,
    window_s: int = 30,
    prefix: int = 32,
    max_prefix: int = 256,
    aging: float = 0.0,
    aging_cap: float = 1.0,
    under_price: bool = True,
) -> Result:
    """Route an arrival stream through fixed market windows.

    With `aging`, every 10 minutes a queued job has waited adds `aging` to its value
    on each platform it bid on, up to `aging_cap` in all: the same credit for every
    job, whatever its size. The federation pays it: the platform gets the whole
    charge, and the job pays the charge less its credit, never below zero.

    With `under_price`, a bid under a platform's price is an offer there too, and wins
    only nodes the auctions leave free, paying its whole bid. Without it, such a bid
    is no offer: the nodes stay free for the next window.
    """
    for name, value in (
        ("window_s", window_s),
        ("prefix", prefix),
        ("max_prefix", max_prefix),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max_prefix < prefix:
        raise ValueError("max_prefix must be at least prefix")
    # A negative credit could withdraw a job's every offer and leave it queued forever.
    if aging < 0 or aging_cap < 0:
        raise ValueError("aging and aging_cap must not be negative")

    # The sort is stable, so jobs submitted together keep their input order.
    ordered = sorted(jobs, key=lambda job: job.submit_s)
    full = {name: platform.exposed_nodes for name, platform in platforms.items()}
    arrivals = []
    waiting = []
    for job in ordered:
        # Nothing is turned away. A job that no platform could run even idle, or
        # that bid on none that could, waits outside the auction until the run ends.
        reason = _blocked_by(job, platforms, full, mechanism)
        if reason is None and not under_price:
            # A job whose bids could never cover a price, even with all the credit
            # aging can give it, would wait in the queue forever.
            most = aging_cap if aging > 0 else 0.0
            if not mechanism.offers(
                _offered(job, platforms, most, False), platforms, full
            ):
                reason = "under_price"
        if reason is None:
            arrivals.append(job)
        else:
            waiting.append(Waiting(job.job_id, reason, job.submit_s))

    routed = []
    queue = []
    arrival = 0
    t = 0
    while arrival < len(arrivals) or queue:
        if not queue:
            # Nothing waits, so nothing changes before the next arrival's window.
            t = max(t, -(-arrivals[arrival].submit_s // window_s) * window_s)
        for platform in platforms.values():
            platform.advance_to(t)
        while arrival < len(arrivals) and arrivals[arrival].submit_s <= t:
            queue.append(arrivals[arrival])
            arrival += 1

        free = {name: platform.free_nodes() for name, platform in platforms.items()}
        # The window opens on public facts: the first jobs that can be placed now,
        # whatever they bid. Later jobs get only the nodes its auctions leave free.
        able = [
            _offered(job, platforms, _credit(job, t, aging, aging_cap), under_price)
            for job in queue
            if candidates(job, platforms, free)
        ]
        winners = _auction(able, platforms, free, mechanism, prefix, max_prefix)
        grouped = {name: [] for name in platforms}
        for job, decision, _, _ in winners:
            grouped[decision.platform].append(job)
        for name, platform in platforms.items():
            if grouped[name]:
                platform.submit(grouped[name], t)

        for platform in platforms.values():
            platform.advance_to(t)
        for name, platform in platforms.items():
            if platform.waiting() != 0:
                raise MarketError(f"{name}: a winner is waiting")

        queued = {job.job_id: job for job in queue}
        for job, decision, cost, value in winners:
            platform = platforms[decision.platform]
            # The mechanism read the credit as part of the bid; the job is worth
            # what it bid, and the federation pays the rest, never past the charge.
            worth = job_value(queued[job.job_id], platform)
            routed.append(
                RoutedJob(
                    job.job_id,
                    decision.platform,
                    t // window_s,
                    cost,
                    worth,
                    decision.charge - cost,
                    decision.charge,
                    min(value - worth, decision.charge),
                    job.submit_s,
                    t,
                    t + platform.run_time(job),
                )
            )
        winner_ids = {decision.job_id for _, decision, _, _ in winners}
        queue = [job for job in queue if job.job_id not in winner_ids]
        t += window_s

    drain = max((row.end_s for row in routed), default=t)
    drain = max(drain, t)
    for platform in platforms.values():
        platform.advance_to(drain)
    statistics = {name: platform.statistics() for name, platform in platforms.items()}
    return Result(
        routed,
        waiting,
        statistics,
        _configuration(
            platforms,
            mechanism,
            window_s,
            prefix,
            max_prefix,
            aging,
            aging_cap,
            under_price,
        ),
        {job.job_id: (job.source, job.num_nodes) for job in ordered},
    )


def _write_csv(path, kind, rows):
    names = [field.name for field in fields(kind)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _service_row(community, jobs) -> _Service:
    """Summarize service for routed rows paired with their node counts."""
    if not jobs:
        return _Service(community, 0, 0.0, 0.0, 0.0)
    waits = [row.begin_s - row.submit_s for row, _ in jobs]
    runs = [row.end_s - row.begin_s for row, _ in jobs]
    weights = [nodes * run / 3600 for (_, nodes), run in zip(jobs, runs)]
    slowdowns = [
        max(1.0, (wait + run) / max(run, 10)) for wait, run in zip(waits, runs)
    ]
    return _Service(
        community,
        len(jobs),
        sum(waits) / len(jobs),
        sum(wait * weight for wait, weight in zip(waits, weights)) / sum(weights),
        sum(slowdowns) / len(jobs),
    )


def _service(result: Result) -> list[_Service]:
    """Return service measures by community and for all routed jobs."""
    grouped = {}
    all_jobs = []
    for row in result.routed:
        community, nodes = result.communities[row.job_id]
        grouped.setdefault(community, []).append((row, nodes))
        all_jobs.append((row, nodes))
    rows = [_service_row(name, grouped[name]) for name in sorted(grouped)]
    return rows + [_service_row("all", all_jobs)]


def write_outputs(result: Result, out_dir) -> dict[str, str]:
    """Write deterministic market outputs and return their paths and hash."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    routed_path = root / "routed.csv"
    waiting_path = root / "waiting.csv"
    service_path = root / "service.csv"
    summary_path = root / "summary.json"
    _write_csv(routed_path, RoutedJob, result.routed)
    _write_csv(waiting_path, Waiting, result.waiting)
    service = _service(result)
    _write_csv(service_path, _Service, service)
    digest = hashlib.sha256(routed_path.read_bytes()).hexdigest()
    summary = {
        "configuration": result.configuration,
        "service": {
            row.community: {
                name: value
                for name, value in asdict(row).items()
                if name != "community"
            }
            for row in service
        },
        "statistics": result.statistics,
        "welfare": sum(row.value - row.cost for row in result.routed),
        "revenue": sum(row.charge for row in result.routed),
        "subsidy": sum(row.subsidy for row in result.routed),
        "routed": len(result.routed),
        "waiting": len(result.waiting),
        "sha256": digest,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "routed": str(routed_path),
        "waiting": str(waiting_path),
        "service": str(service_path),
        "summary": str(summary_path),
        "sha256": digest,
    }


__all__ = [
    "MarketError",
    "Result",
    "RoutedJob",
    "Waiting",
    "run",
    "write_outputs",
]
