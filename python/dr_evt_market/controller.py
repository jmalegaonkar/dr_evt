################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Run fixed-window market clearing over opaque DR_EVT platform sessions."""

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field
import hashlib
from itertools import product
import json
from math import isfinite
from pathlib import Path

from .mechanisms import (
    JobBid,
    MarketObservation,
    Mechanism,
    QueuedJob,
    RejectedDecision,
    build_observation,
    submit_decisions,
    validate_decisions,
)
from .platforms.base import JobTiming, PlatformReport, PlatformSession


@dataclass(frozen=True)
class RoutedLeg:
    """Record one routed leg and its final platform timing."""

    job_id: str
    leg_id: str
    platform: str
    window_index: int
    window_time_s: int
    handle: int
    value_credits: float
    resource_cost_credits: float
    charge_credits: float
    submit_s: int
    begin_s: int
    end_s: int


@dataclass(frozen=True)
class WindowRecord:
    """Summarize one market clearing window in deterministic order."""

    index: int
    time_s: int
    queued: tuple[str, ...]
    placed: tuple[str, ...]
    rejected_decisions: tuple[RejectedDecision, ...]
    welfare: float
    revenue: float
    free_nodes: dict[str, int]


@dataclass(frozen=True)
class RejectedJob:
    """Record a job rejected by intake or terminal clearing."""

    job_id: str
    reason: str
    time_s: int


@dataclass
class RunReport:
    """Collect routes, windows, rejections, and final platform reports."""

    routed: list[RoutedLeg]
    windows: list[WindowRecord]
    rejected: list[RejectedJob]
    reports: dict[str, PlatformReport]
    configuration: dict[str, object] = field(default_factory=dict)


class RoutingInvariantViolation(RuntimeError):
    """Report a routed leg that did not start at its clearing window."""


@dataclass(frozen=True)
class _PendingRoute:
    job_id: str
    leg_id: str
    platform: str
    window_index: int
    window_time_s: int
    handle: int
    value_credits: float
    resource_cost_credits: float
    charge_credits: float
    submit_s: int


class Controller:
    """Drive deterministic fixed-window clearing across named platforms."""

    def __init__(
        self,
        platforms: Mapping[str, PlatformSession],
        prices: Mapping[str, float],
        mechanism: Mechanism,
        jobs: Sequence[QueuedJob],
        bids: Mapping[str, JobBid],
        *,
        window_s: int,
        seed: int = 0,
    ) -> None:
        """Validate and retain one complete market run configuration."""
        if not platforms:
            raise ValueError("at least one platform is required")
        if (
            not isinstance(window_s, int)
            or isinstance(window_s, bool)
            or window_s < 1
        ):
            raise ValueError("window_s must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")

        self.platforms = dict(sorted(platforms.items()))
        self.prices = {}
        for name, platform in self.platforms.items():
            if platform.name != name:
                raise ValueError(
                    f"platform {name!r} reports name {platform.name!r}"
                )
            try:
                price = float(prices[name])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"platform {name!r} has no valid price") from error
            if not isfinite(price) or price < 0.0:
                raise ValueError(
                    f"platform {name!r} price must be finite and non-negative"
                )
            self.prices[name] = price

        indexed_jobs = list(enumerate(jobs))
        indexed_jobs.sort(key=lambda item: (item[1].submit_s, item[0]))
        self.jobs = tuple(job for _, job in indexed_jobs)
        if len({job.job_id for job in self.jobs}) != len(self.jobs):
            raise ValueError("job IDs must be unique")
        self.bids = dict(bids)
        for job in self.jobs:
            bid = self.bids.get(job.job_id)
            if bid is None:
                raise ValueError(f"job {job.job_id!r} has no bid")
            if tuple(leg.leg_id for leg in bid.legs) != tuple(
                leg.leg_id for leg in job.legs
            ):
                raise ValueError(f"job {job.job_id!r} bid legs do not match")

        self.mechanism = mechanism
        self.window_s = window_s
        self.seed = seed
        self._report: RunReport | None = None

    def _intake_reason(self, job: QueuedJob) -> str | None:
        bid = self.bids[job.job_id]
        bids_by_leg = {leg.leg_id: leg for leg in bid.legs}
        platform_names = tuple(self.platforms)
        choices = [
            tuple(
                name
                for name in platform_names
                if name in bids_by_leg[leg.leg_id].value_by_platform
                and leg.num_nodes <= self.platforms[name].total_nodes
            )
            for leg in job.legs
        ]
        if any(not leg_choices for leg_choices in choices):
            return "oversize"

        structurally_feasible = False
        for assignment in product(*choices):
            demand = {}
            value = 0.0
            cost = 0.0
            for leg, platform in zip(job.legs, assignment):
                demand[platform] = demand.get(platform, 0) + leg.num_nodes
                value += bids_by_leg[leg.leg_id].value_by_platform[platform]
                cost += (
                    self.prices[platform]
                    * leg.num_nodes
                    * leg.limit_s
                    / 3600.0
                )
            if any(
                nodes > self.platforms[name].total_nodes
                for name, nodes in demand.items()
            ):
                continue
            structurally_feasible = True
            if value >= cost:
                return None
        return "unaffordable" if structurally_feasible else "oversize"

    def run(self) -> RunReport:
        """Run fixed clearing windows through drain and return final records."""
        if self._report is not None:
            return self._report

        arrivals = []
        rejected_jobs = []
        for job in self.jobs:
            reason = self._intake_reason(job)
            if reason is None:
                arrivals.append(job)
            else:
                rejected_jobs.append(RejectedJob(
                    job.job_id,
                    reason,
                    job.submit_s,
                ))

        queued: list[QueuedJob] = []
        pending_routes: list[_PendingRoute] = []
        windows = []
        arrival_index = 0
        window_index = 0
        time_s = 0
        while arrival_index < len(arrivals) or queued:
            for platform in self.platforms.values():
                platform.advance_to(time_s)
            while (
                arrival_index < len(arrivals)
                and arrivals[arrival_index].submit_s <= time_s
            ):
                queued.append(arrivals[arrival_index])
                arrival_index += 1

            snapshots = {
                name: platform.snapshot()
                for name, platform in self.platforms.items()
            }
            observation = build_observation(
                time_s,
                window_index,
                self.seed,
                queued,
                self.bids,
                snapshots,
                self.prices,
            )
            decisions = self.mechanism.decide(observation)
            accepted, rejected_decisions = validate_decisions(
                observation,
                decisions,
            )
            handles = submit_decisions(
                accepted,
                observation,
                self.platforms,
                time_s,
            )
            for platform in self.platforms.values():
                platform.advance_to(time_s)

            timings = self._new_timings(accepted, observation, handles)
            for decision in accepted:
                offer = observation.job(decision.job_id)
                candidate = observation.candidate(
                    decision.job_id,
                    decision.placement_id,
                )
                value = observation.value(
                    decision.job_id,
                    decision.placement_id,
                )
                if value is None:
                    raise RoutingInvariantViolation(
                        f"accepted job {decision.job_id!r} has no value"
                    )
                for leg in offer.legs:
                    key = (decision.job_id, leg.leg_id)
                    platform_name = candidate.platform_by_leg[leg.leg_id]
                    timing = timings[key]
                    if not timing.scheduled or timing.begin_s != float(time_s):
                        raise RoutingInvariantViolation(
                            f"leg {decision.job_id}/{leg.leg_id} began at "
                            f"{timing.begin_s}, not window time {time_s}"
                        )
                    pending_routes.append(_PendingRoute(
                        decision.job_id,
                        leg.leg_id,
                        platform_name,
                        window_index,
                        time_s,
                        handles[key],
                        value,
                        candidate.resource_cost_credits,
                        decision.charge_credits,
                        offer.submit_s,
                    ))

            queued_ids = tuple(job.job_id for job in queued)
            placed_ids = tuple(decision.job_id for decision in accepted)
            welfare = sum(
                observation.net_value(
                    decision.job_id,
                    decision.placement_id,
                ) or 0.0
                for decision in accepted
            )
            windows.append(WindowRecord(
                window_index,
                time_s,
                queued_ids,
                placed_ids,
                tuple(rejected_decisions),
                welfare,
                sum(decision.charge_credits for decision in accepted),
                {
                    name: snapshot.free_nodes
                    for name, snapshot in snapshots.items()
                },
            ))
            placed = set(placed_ids)
            queued = [job for job in queued if job.job_id not in placed]
            if (
                not accepted
                and queued
                and all(
                    snapshot.free_nodes == snapshot.total_nodes
                    for snapshot in snapshots.values()
                )
            ):
                rejected_jobs.extend(
                    RejectedJob(job.job_id, "unplaceable", time_s)
                    for job in queued
                )
                queued.clear()
                break
            window_index += 1
            time_s += self.window_s

        reports = {
            name: platform.finish()
            for name, platform in self.platforms.items()
        }
        routed = self._final_routes(pending_routes, reports)
        configuration = {
            "mechanism": self.mechanism.name,
            "seed": self.seed,
            "window_s": self.window_s,
            "platforms": {
                name: {
                    "total_nodes": platform.total_nodes,
                    "price_per_node_hour": self.prices[name],
                }
                for name, platform in self.platforms.items()
            },
        }
        self._report = RunReport(
            routed,
            windows,
            rejected_jobs,
            reports,
            configuration,
        )
        return self._report

    def _new_timings(
        self,
        decisions: Sequence,
        observation: MarketObservation,
        handles: Mapping[tuple[str, str], int],
    ) -> dict[tuple[str, str], JobTiming]:
        keys_by_platform: dict[str, list[tuple[str, str]]] = {
            name: [] for name in self.platforms
        }
        for decision in decisions:
            offer = observation.job(decision.job_id)
            candidate = observation.candidate(
                decision.job_id,
                decision.placement_id,
            )
            for leg in offer.legs:
                platform = candidate.platform_by_leg[leg.leg_id]
                keys_by_platform[platform].append((decision.job_id, leg.leg_id))

        result = {}
        for name, keys in keys_by_platform.items():
            if not keys:
                continue
            platform_handles = [handles[key] for key in keys]
            platform_timings = self.platforms[name].timings(platform_handles)
            result.update(zip(keys, platform_timings))
        return result

    @staticmethod
    def _final_routes(
        pending: Sequence[_PendingRoute],
        reports: Mapping[str, PlatformReport],
    ) -> list[RoutedLeg]:
        timings = {
            (name, timing.handle): timing
            for name, report in reports.items()
            for timing in report.timings
        }
        routed = []
        for route in pending:
            timing = timings[(route.platform, route.handle)]
            routed.append(RoutedLeg(
                route.job_id,
                route.leg_id,
                route.platform,
                route.window_index,
                route.window_time_s,
                route.handle,
                route.value_credits,
                route.resource_cost_credits,
                route.charge_credits,
                route.submit_s,
                _integer_time(timing.begin_s, "begin_s"),
                _integer_time(timing.end_s, "end_s"),
            ))
        return routed


def _integer_time(value: float, field_name: str) -> int:
    if not float(value).is_integer():
        raise RoutingInvariantViolation(
            f"{field_name} must be an integer number of seconds, got {value}"
        )
    return int(value)


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: Sequence[Mapping[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(
    report: RunReport,
    out_dir: str | Path,
) -> dict[str, str]:
    """Write deterministic route, window, rejection, and manifest outputs."""
    directory = Path(out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    routed_path = directory / "routed.csv"
    windows_path = directory / "windows.csv"
    rejected_path = directory / "rejected.csv"
    manifest_path = directory / "run.json"

    routed_fields = (
        "job_id",
        "leg_id",
        "platform",
        "window_index",
        "window_time_s",
        "handle",
        "value_credits",
        "resource_cost_credits",
        "charge_credits",
        "submit_s",
        "begin_s",
        "end_s",
    )
    _write_csv(
        routed_path,
        routed_fields,
        [
            {
                field_name: getattr(route, field_name)
                for field_name in routed_fields
            }
            for route in report.routed
        ],
    )

    window_fields = (
        "index",
        "time_s",
        "queued",
        "placed",
        "rejected_decisions",
        "welfare",
        "revenue",
        "free_nodes",
    )
    _write_csv(
        windows_path,
        window_fields,
        [
            {
                "index": window.index,
                "time_s": window.time_s,
                "queued": json.dumps(list(window.queued), separators=(",", ":")),
                "placed": json.dumps(list(window.placed), separators=(",", ":")),
                "rejected_decisions": json.dumps(
                    [
                        {
                            "job_id": item.decision.job_id,
                            "placement_id": item.decision.placement_id,
                            "reason": item.reason,
                        }
                        for item in window.rejected_decisions
                    ],
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "welfare": window.welfare,
                "revenue": window.revenue,
                "free_nodes": json.dumps(
                    window.free_nodes,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
            for window in report.windows
        ],
    )
    _write_csv(
        rejected_path,
        ("job_id", "reason", "time_s"),
        [
            {
                "job_id": rejected.job_id,
                "reason": rejected.reason,
                "time_s": rejected.time_s,
            }
            for rejected in report.rejected
        ],
    )

    hashes = {
        "routed.csv": _sha256(routed_path),
        "windows.csv": _sha256(windows_path),
    }
    manifest = {
        "configuration": report.configuration,
        "platform_statistics": {
            name: platform_report.statistics
            for name, platform_report in sorted(report.reports.items())
        },
        "sha256": hashes,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "routed": str(routed_path),
        "windows": str(windows_path),
        "rejected": str(rejected_path),
        "manifest": str(manifest_path),
        "routed_sha256": hashes["routed.csv"],
        "windows_sha256": hashes["windows.csv"],
    }
