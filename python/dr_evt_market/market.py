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
from dataclasses import asdict, dataclass
from pathlib import Path

from .mechanism import Decision, Mechanism, candidates

_TOLERANCE = 1.0e-9
_ROUTED_FIELDS = (
    "job_id",
    "platform",
    "window",
    "window_s",
    "cost",
    "value",
    "premium",
    "charge",
    "submit_s",
    "begin_s",
    "end_s",
)
_WAITING_FIELDS = ("job_id", "reason", "submit_s")
_SERVICE_FIELDS = (
    "community",
    "count",
    "mean_wait_s",
    "node_hour_weighted_mean_wait_s",
    "mean_bounded_slowdown",
)


@dataclass(frozen=True)
class RoutedJob:
    """One winning job and its realized route and payment."""

    job_id: str
    platform: str
    window: int
    window_s: int
    cost: float
    value: float
    premium: float
    charge: float
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


def _check_decisions(batch, platforms, free_nodes, decisions: list[Decision]):
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
        offer = candidates(job, platforms, free_nodes).get(decision.platform)
        if offer is None:
            raise MarketError(f"{decision.job_id}: platform is not a candidate")
        cost, value = offer
        if not cost - _TOLERANCE <= decision.charge <= value + _TOLERANCE:
            raise MarketError(f"{decision.job_id}: charge is outside its offer")
        used[decision.platform] += job.num_nodes
        if used[decision.platform] > free_nodes[decision.platform]:
            raise MarketError(f"{decision.job_id}: platform capacity exceeded")
        checked[decision.job_id] = (job, decision, cost, value)
    left = {name: free_nodes[name] - used[name] for name in platforms}
    for job in batch:
        if job.job_id not in checked and candidates(job, platforms, left):
            raise MarketError(f"{job.job_id}: left waiting on free nodes")
    return [checked[job.job_id] for job in batch if job.job_id in checked]


def _configuration(platforms, mechanism, window_s, prefix):
    return {
        "mechanism": mechanism.name,
        "window_s": window_s,
        "prefix": prefix,
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


def _blocked_by(job, platforms, full_nodes):
    if not any(job.requires <= platform.hardware for platform in platforms.values()):
        return "hardware"
    if not any(platform.fits(job) for platform in platforms.values()):
        return "oversize"
    if not candidates(job, platforms, full_nodes):
        return "unaffordable"
    return None


def run(
    jobs, platforms, mechanism: Mechanism, *, window_s: int = 60, prefix: int = 32
) -> Result:
    """Route an arrival stream through fixed market windows."""
    if isinstance(window_s, bool) or not isinstance(window_s, int) or window_s <= 0:
        raise ValueError("window_s must be a positive integer")
    if isinstance(prefix, bool) or not isinstance(prefix, int) or prefix <= 0:
        raise ValueError("prefix must be a positive integer")

    ordered = [
        job
        for _, job in sorted(
            enumerate(jobs), key=lambda item: (item[1].submit_s, item[0])
        )
    ]
    full = {name: platform.exposed_nodes for name, platform in platforms.items()}
    arrivals = []
    waiting = []
    for job in ordered:
        # Nothing is turned away. A job that no platform could run at its price,
        # even idle, waits outside the auction: at fixed prices and shares, until
        # the run ends.
        reason = _blocked_by(job, platforms, full)
        if reason is None:
            arrivals.append(job)
        else:
            waiting.append(Waiting(job.job_id, reason, job.submit_s))

    routed = []
    queue = []
    arrival = 0
    t = 0
    window = 0
    while arrival < len(arrivals) or queue:
        for platform in platforms.values():
            platform.advance_to(t)
        while arrival < len(arrivals) and arrivals[arrival].submit_s <= t:
            queue.append(arrivals[arrival])
            arrival += 1

        free = {name: platform.free_nodes() for name, platform in platforms.items()}
        batch = []
        for job in queue:
            if candidates(job, platforms, free):
                batch.append(job)
                if len(batch) == prefix:
                    break
        decisions = list(mechanism.decide(batch, platforms, free))
        winners = _check_decisions(batch, platforms, free, decisions)
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

        for job, decision, cost, value in winners:
            routed.append(
                RoutedJob(
                    job.job_id,
                    decision.platform,
                    window,
                    t,
                    cost,
                    value,
                    decision.charge - cost,
                    decision.charge,
                    job.submit_s,
                    t,
                    t + platforms[decision.platform].run_time(job),
                )
            )
        winner_ids = {decision.job_id for decision in decisions}
        queue = [job for job in queue if job.job_id not in winner_ids]
        t += window_s
        window += 1

    drain = max((row.end_s for row in routed), default=t)
    drain = max(drain, t)
    for platform in platforms.values():
        platform.advance_to(drain)
    statistics = {name: platform.statistics() for name, platform in platforms.items()}
    return Result(
        routed,
        waiting,
        statistics,
        _configuration(platforms, mechanism, window_s, prefix),
        {job.job_id: (job.source, job.num_nodes) for job in ordered},
    )


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
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
    _write_csv(routed_path, _ROUTED_FIELDS, result.routed)
    _write_csv(waiting_path, _WAITING_FIELDS, result.waiting)
    service = _service(result)
    _write_csv(service_path, _SERVICE_FIELDS, service)
    digest = hashlib.sha256(routed_path.read_bytes()).hexdigest()
    summary = {
        "configuration": result.configuration,
        "service": {
            row.community: {field: getattr(row, field) for field in _SERVICE_FIELDS[1:]}
            for row in service
        },
        "statistics": result.statistics,
        "welfare": sum(row.value - row.cost for row in result.routed),
        "revenue": sum(row.charge for row in result.routed),
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
