################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The loop: take the queue, run the auction, hand winners to their platforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from .mechanisms import (
    Job,
    Mechanism,
    Platform,
    Rejection,
    Window,
    build_window,
    placements,
    submit,
    validate_decisions,
)
from .platforms.base import JobTiming, PlatformReport, PlatformSession


@dataclass(frozen=True)
class RoutedLeg:
    """One leg that ran: where it went, what the job paid, what happened."""

    job_id: str
    leg_id: str
    platform: str
    window_index: int
    window_time_s: int
    handle: int
    cost_credits: float
    value_credits: float
    premium_credits: float
    charge_credits: float
    submit_s: int
    begin_s: int
    end_s: int


@dataclass(frozen=True)
class WindowRecord:
    """What one window saw and did."""

    index: int
    time_s: int
    queued: tuple[str, ...]
    placed: tuple[str, ...]
    rejected: tuple[Rejection, ...]
    welfare: float
    revenue: float
    free_nodes: dict[str, int]


@dataclass
class RunReport:
    """Everything a run produced."""

    routed: list[RoutedLeg]
    windows: list[WindowRecord]
    rejected: list[Rejection]
    reports: dict[str, PlatformReport]
    configuration: dict[str, object] = field(default_factory=dict)


class RoutingError(RuntimeError):
    """A platform did not start a winner at its window."""


class Controller:
    """Clear the market at fixed boundaries until the job stream is drained."""

    def __init__(
        self,
        sessions: Mapping[str, PlatformSession],
        platforms: Mapping[str, Platform],
        mechanism: Mechanism,
        jobs: Sequence[Job],
        *,
        window_s: int,
        seed: int = 0,
        log_dir: str | Path | None = None,
    ) -> None:
        """Check that sessions and platforms agree and keep the run settings."""
        if not sessions or set(sessions) != set(platforms):
            raise ValueError("sessions and platforms must name the same clusters")
        for name, session in sessions.items():
            if (
                session.name != name
                or session.total_nodes != platforms[name].total_nodes
            ):
                raise ValueError(f"session {name!r} does not match its platform")
        if isinstance(window_s, bool) or not isinstance(window_s, int) or window_s < 1:
            raise ValueError("window_s must be a positive integer")
        self.sessions = dict(sorted(sessions.items()))
        self.platforms = dict(sorted(platforms.items()))
        self.mechanism = mechanism
        ordered = sorted(enumerate(jobs), key=lambda item: (item[1].submit_s, item[0]))
        self.jobs = tuple(job for _, job in ordered)
        if len({job.job_id for job in self.jobs}) != len(self.jobs):
            raise ValueError("job ids must be unique")
        self.window_s = window_s
        self.seed = seed
        self.log_dir = Path(log_dir).resolve() if log_dir is not None else None
        self._report: RunReport | None = None

    def _intake(self, job: Job) -> str | None:
        options = placements(job, self.platforms)
        if not options:
            return "oversize"
        if all(option.value_credits < option.cost_credits for option in options):
            return "unaffordable"
        return None

    def run(self) -> RunReport:
        """Run every window through the drain and return the records."""
        if self._report is not None:
            return self._report
        arrivals = []
        rejected = []
        for job in self.jobs:
            reason = self._intake(job)
            if reason is None:
                arrivals.append(job)
            else:
                rejected.append(Rejection(job.job_id, reason, job.submit_s))

        queue: list[Job] = []
        pending: list[tuple] = []
        windows: list[WindowRecord] = []
        next_arrival = 0
        index = 0
        time_s = 0
        while next_arrival < len(arrivals) or queue:
            for session in self.sessions.values():
                session.advance_to(time_s)
            while (
                next_arrival < len(arrivals)
                and arrivals[next_arrival].submit_s <= time_s
            ):
                queue.append(arrivals[next_arrival])
                next_arrival += 1
            snapshots = {
                name: session.snapshot() for name, session in self.sessions.items()
            }
            window = build_window(
                time_s,
                index,
                queue,
                self.platforms,
                {name: snapshot.free_nodes for name, snapshot in snapshots.items()},
            )
            if self.log_dir is not None:
                _write_window(self.log_dir, window)
            accepted, refused = validate_decisions(
                window, self.mechanism.decide(window)
            )
            handles = submit(accepted, window, self.sessions, time_s)
            for session in self.sessions.values():
                session.advance_to(time_s)
            timings = self._timings(accepted, window, handles)
            for decision in accepted:
                job = window.job(decision.job_id)
                for leg, name in zip(job.legs, decision.placement.platforms):
                    timing = timings[(job.job_id, leg.leg_id)]
                    if not timing.scheduled or timing.begin_s != float(time_s):
                        raise RoutingError(
                            f"{job.job_id}/{leg.leg_id} began at {timing.begin_s}, "
                            f"not at its window {time_s}"
                        )
                    pending.append(
                        (
                            decision,
                            job,
                            leg,
                            name,
                            index,
                            time_s,
                            handles[(job.job_id, leg.leg_id)],
                        )
                    )
            placed = tuple(decision.job_id for decision in accepted)
            windows.append(
                WindowRecord(
                    index,
                    time_s,
                    tuple(job.job_id for job in queue),
                    placed,
                    tuple(refused),
                    sum(decision.placement.net_credits for decision in accepted),
                    sum(decision.charge_credits for decision in accepted),
                    {name: snapshot.free_nodes for name, snapshot in snapshots.items()},
                )
            )
            queue = [job for job in queue if job.job_id not in set(placed)]
            if (
                not accepted
                and queue
                and all(
                    snapshot.free_nodes == snapshot.total_nodes
                    for snapshot in snapshots.values()
                )
            ):
                rejected.extend(
                    Rejection(job.job_id, "unplaceable", time_s) for job in queue
                )
                queue.clear()
                break
            index += 1
            time_s += self.window_s

        reports = {name: session.finish() for name, session in self.sessions.items()}
        final = {
            (name, timing.handle): timing
            for name, report in reports.items()
            for timing in report.timings
        }
        routed = []
        for decision, job, leg, name, window_index, window_time, handle in pending:
            timing = final[(name, handle)]
            routed.append(
                RoutedLeg(
                    job.job_id,
                    leg.leg_id,
                    name,
                    window_index,
                    window_time,
                    handle,
                    decision.placement.cost_credits,
                    decision.placement.value_credits,
                    decision.charge_credits - decision.placement.cost_credits,
                    decision.charge_credits,
                    job.submit_s,
                    _whole_seconds(timing.begin_s),
                    _whole_seconds(timing.end_s),
                )
            )
        self._report = RunReport(
            routed,
            windows,
            rejected,
            reports,
            {
                "mechanism": self.mechanism.name,
                "seed": self.seed,
                "window_s": self.window_s,
                "platforms": {
                    name: {
                        "total_nodes": platform.total_nodes,
                        "price_per_node_hour": platform.price_per_node_hour,
                        "hardware": sorted(platform.hardware),
                    }
                    for name, platform in self.platforms.items()
                },
            },
        )
        return self._report

    def _timings(
        self,
        accepted: Sequence,
        window: Window,
        handles: Mapping[tuple[str, str], int],
    ) -> dict[tuple[str, str], JobTiming]:
        keys: dict[str, list[tuple[str, str]]] = {name: [] for name in self.sessions}
        for decision in accepted:
            job = window.job(decision.job_id)
            for leg, name in zip(job.legs, decision.placement.platforms):
                keys[name].append((job.job_id, leg.leg_id))
        result = {}
        for name, names in keys.items():
            if names:
                records = self.sessions[name].timings([handles[key] for key in names])
                result.update(zip(names, records))
        return result


def _whole_seconds(value: float) -> int:
    if not float(value).is_integer():
        raise RoutingError(f"expected whole seconds, got {value}")
    return int(value)


def _write_window(directory: Path, window: Window) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "time_s": window.time_s,
        "index": window.index,
        "free_nodes": dict(window.free_nodes),
        "platforms": {
            name: {
                "total_nodes": platform.total_nodes,
                "price_per_node_hour": platform.price_per_node_hour,
                "hardware": sorted(platform.hardware),
            }
            for name, platform in window.platforms.items()
        },
        "jobs": [
            {
                "job_id": job.job_id,
                "submit_s": job.submit_s,
                "bid": job.bid,
                "legs": [
                    {
                        "leg_id": leg.leg_id,
                        "num_nodes": leg.num_nodes,
                        "limit_s": leg.limit_s,
                        "requires": sorted(leg.requires),
                    }
                    for leg in job.legs
                ],
            }
            for job in window.jobs
        ],
    }
    (directory / f"window_{window.index:06d}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, fields: tuple[str, ...], rows: Sequence[Mapping]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(report: RunReport, out_dir: str | Path) -> dict[str, str]:
    """Write routed.csv, windows.csv, rejected.csv and run.json with hashes."""
    directory = Path(out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    routed_fields = tuple(RoutedLeg.__dataclass_fields__)
    _write_csv(
        directory / "routed.csv",
        routed_fields,
        [{name: getattr(leg, name) for name in routed_fields} for leg in report.routed],
    )
    _write_csv(
        directory / "windows.csv",
        (
            "index",
            "time_s",
            "queued",
            "placed",
            "rejected",
            "welfare",
            "revenue",
            "free_nodes",
        ),
        [
            {
                "index": window.index,
                "time_s": window.time_s,
                "queued": json.dumps(list(window.queued), separators=(",", ":")),
                "placed": json.dumps(list(window.placed), separators=(",", ":")),
                "rejected": json.dumps(
                    [{"job_id": r.job_id, "reason": r.reason} for r in window.rejected],
                    separators=(",", ":"),
                ),
                "welfare": window.welfare,
                "revenue": window.revenue,
                "free_nodes": json.dumps(
                    window.free_nodes, separators=(",", ":"), sort_keys=True
                ),
            }
            for window in report.windows
        ],
    )
    _write_csv(
        directory / "rejected.csv",
        ("job_id", "reason", "time_s"),
        [
            {"job_id": r.job_id, "reason": r.reason, "time_s": r.time_s}
            for r in report.rejected
        ],
    )
    hashes = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("routed.csv", "windows.csv")
    }
    (directory / "run.json").write_text(
        json.dumps(
            {
                "configuration": report.configuration,
                "platform_statistics": {
                    name: report.statistics
                    for name, report in sorted(report.reports.items())
                },
                "sha256": hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "routed": str(directory / "routed.csv"),
        "windows": str(directory / "windows.csv"),
        "rejected": str(directory / "rejected.csv"),
        "manifest": str(directory / "run.json"),
        "routed_sha256": hashes["routed.csv"],
        "windows_sha256": hashes["windows.csv"],
    }


__all__ = [
    "Controller",
    "RoutedLeg",
    "RoutingError",
    "RunReport",
    "WindowRecord",
    "write_outputs",
]
