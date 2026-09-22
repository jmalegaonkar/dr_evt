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
_REJECTED_FIELDS = ("job_id", "reason", "time_s")


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
class Rejected:
    """One job rejected by intake or the market loop."""

    job_id: str
    reason: str
    time_s: int


@dataclass
class Result:
    """The routes, rejections, statistics and resolved configuration."""

    routed: list[RoutedJob]
    rejected: list[Rejected]
    statistics: dict[str, dict[str, float]]
    configuration: dict


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
            }
            for name, platform in platforms.items()
        },
    }


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
    rejected = []
    for job in ordered:
        if not any(platform.fits(job) for platform in platforms.values()):
            rejected.append(Rejected(job.job_id, "oversize", job.submit_s))
        elif not candidates(job, platforms, full):
            rejected.append(Rejected(job.job_id, "unaffordable", job.submit_s))
        else:
            arrivals.append(job)

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

        batch = queue[:prefix]
        free = {name: platform.free_nodes() for name, platform in platforms.items()}
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
                    t + job.limit_s,
                )
            )
        winner_ids = {decision.job_id for decision in decisions}
        queue = [job for job in queue if job.job_id not in winner_ids]

        if (
            not winners
            and queue
            and all(
                platform.free_nodes() == platform.exposed_nodes
                for platform in platforms.values()
            )
        ):
            rejected.extend(Rejected(job.job_id, "unplaceable", t) for job in queue)
            queue.clear()
            break
        t += window_s
        window += 1

    drain = max((row.end_s for row in routed), default=t)
    drain = max(drain, t)
    for platform in platforms.values():
        platform.advance_to(drain)
    statistics = {name: platform.statistics() for name, platform in platforms.items()}
    return Result(
        routed,
        rejected,
        statistics,
        _configuration(platforms, mechanism, window_s, prefix),
    )


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_outputs(result: Result, out_dir) -> dict[str, str]:
    """Write deterministic market outputs and return their paths and hash."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    routed_path = root / "routed.csv"
    rejected_path = root / "rejected.csv"
    summary_path = root / "summary.json"
    _write_csv(routed_path, _ROUTED_FIELDS, result.routed)
    _write_csv(rejected_path, _REJECTED_FIELDS, result.rejected)
    digest = hashlib.sha256(routed_path.read_bytes()).hexdigest()
    summary = {
        "configuration": result.configuration,
        "statistics": result.statistics,
        "welfare": sum(row.value - row.cost for row in result.routed),
        "revenue": sum(row.charge for row in result.routed),
        "routed": len(result.routed),
        "rejected": len(result.rejected),
        "sha256": digest,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "routed": str(routed_path),
        "rejected": str(rejected_path),
        "summary": str(summary_path),
        "sha256": digest,
    }


__all__ = [
    "MarketError",
    "Rejected",
    "Result",
    "RoutedJob",
    "run",
    "write_outputs",
]
