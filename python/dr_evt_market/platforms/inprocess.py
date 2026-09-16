################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""In-process implementation of the market platform adapter contract."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import (
    _DRAIN_TIME_S,
    _STATISTIC_FIELDS,
    _is_integer,
    ClockViolation,
    ConfigurationError,
    InfrastructureFailure,
    JobTiming,
    PlatformReport,
    PlatformSnapshot,
    StructuralRejection,
    SubmitRequest,
    validate,
)

_DR_EVT_ERRORS = (RuntimeError, ValueError)


class InProcessPlatform:
    """Drive one DR_EVT simulation in the current Python process."""

    def __init__(
        self,
        name: str,
        total_nodes: int,
        work_dir: str | Path,
        *,
        backfill: str = "easy",
    ) -> None:
        """Create an EASY/FCFS platform and its output files."""
        if not isinstance(name, str) or not name:
            raise ConfigurationError("platform name must be a non-empty string")
        if not _is_integer(total_nodes) or total_nodes < 1:
            raise ConfigurationError("total_nodes must be a positive integer")
        if backfill != "easy":
            raise ConfigurationError("backfill must be 'easy'")

        try:
            import dr_evt
        except ImportError as error:
            raise InfrastructureFailure(
                "the dr_evt Python extension is not installed"
            ) from error

        self.name = name
        self.total_nodes = total_nodes
        self._work_dir = Path(work_dir).resolve()
        try:
            self._work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise InfrastructureFailure(
                f"cannot create platform work directory {self._work_dir}"
            ) from error

        self._header_path = self._work_dir / f"{name}.header.csv"
        self._simulated_trace_path = (
            self._work_dir / f"{name}.simulated.csv"
        )
        self._resource_trace_path = self._work_dir / f"{name}.resource.csv"
        try:
            self._header_path.write_text(
                "job_submit_time,num_nodes,time_limit\n",
                encoding="utf-8",
            )
        except OSError as error:
            raise InfrastructureFailure(
                f"cannot write platform header {self._header_path}"
            ) from error

        try:
            params = dr_evt.SimParams()
            params.infile = str(self._header_path)
            params.total_nodes = total_nodes
            params.trace_format = "simple"
            params.timestamp_format = "epoch"
            params.run_time_mode = dr_evt.RunTimeMode.LIMIT
            params.backfill_policy = dr_evt.BackfillPolicy.EASY
            params.priority_policy = dr_evt.PriorityPolicy.FCFS
            params.msec_output = True
            params.outfile = str(self._simulated_trace_path)
            params.resource_trace = str(self._resource_trace_path)
            self._simulation = dr_evt.Simulation(params)
        except _DR_EVT_ERRORS as error:
            raise InfrastructureFailure(
                f"cannot initialize platform {name}: {error}"
            ) from error

        self._params = params
        self._dr_evt: Any = dr_evt
        self._ledger: dict[int, str] = {}
        self._handles: list[int] = []
        self._last_submit_s: int | None = None
        self._report: PlatformReport | None = None

    def now(self) -> int:
        """Return the platform's current integer simulation time."""
        self._ensure_open()
        try:
            return int(self._simulation.get_current_time())
        except _DR_EVT_ERRORS as error:
            raise InfrastructureFailure(
                f"cannot read platform {self.name!r} time: {error}"
            ) from error

    def _ensure_open(self) -> None:
        if self._report is not None:
            raise InfrastructureFailure(
                f"platform {self.name!r} is finished"
            )

    def submit(self, jobs: Sequence[SubmitRequest]) -> list[int]:
        """Validate and append one chronologically ordered request batch."""
        self._ensure_open()
        requests = list(jobs)
        previous_submit = self._last_submit_s
        current_time = self.now()
        for request in requests:
            validate(request)
            if request.num_nodes > self.total_nodes:
                raise StructuralRejection(
                    f"job {request.key!r} requests {request.num_nodes} nodes, "
                    f"but platform {self.name!r} has {self.total_nodes}"
                )
            if request.submit_s < current_time:
                raise ClockViolation(
                    f"job {request.key!r} submits at {request.submit_s}, "
                    f"before platform time {current_time}"
                )
            if (previous_submit is not None
                    and request.submit_s < previous_submit):
                raise ClockViolation(
                    f"job {request.key!r} submits at {request.submit_s}, "
                    f"before the previous submission at {previous_submit}"
                )
            previous_submit = request.submit_s

        if not requests:
            return []

        try:
            append_requests = [
                self._dr_evt.JobAppendRequest(
                    request.submit_s,
                    request.num_nodes,
                    request.q_id,
                    request.limit_s,
                )
                for request in requests
            ]
            handles = [
                int(handle)
                for handle in self._simulation.append_jobs(append_requests)
            ]
        except _DR_EVT_ERRORS as error:
            raise InfrastructureFailure(
                f"platform {self.name!r} rejected an append batch: {error}"
            ) from error
        if len(handles) != len(requests):
            raise InfrastructureFailure(
                f"platform {self.name!r} returned {len(handles)} handles "
                f"for {len(requests)} requests"
            )

        for handle, request in zip(handles, requests):
            self._ledger[handle] = request.key
            self._handles.append(handle)
        self._last_submit_s = previous_submit
        return handles

    def advance_to(self, time_s: int) -> None:
        """Advance through events at time_s after enforcing a monotone clock."""
        self._ensure_open()
        if not _is_integer(time_s):
            raise ClockViolation("advance time must be an integer number of seconds")
        current_time = self.now()
        if time_s < current_time:
            raise ClockViolation(
                f"cannot advance platform {self.name!r} from "
                f"{current_time} back to {time_s}"
            )
        try:
            self._simulation.advance_to(time_s)
        except _DR_EVT_ERRORS as error:
            raise InfrastructureFailure(
                f"cannot advance platform {self.name!r}: {error}"
            ) from error

    def snapshot(self) -> PlatformSnapshot:
        """Return capacity, queue, and utilization state."""
        self._ensure_open()
        try:
            return PlatformSnapshot(
                name=self.name,
                time_s=self.now(),
                total_nodes=self.total_nodes,
                free_nodes=int(self._simulation.get_available_nodes()),
                in_use_nodes=int(self._simulation.get_nodes_in_use()),
                waiting_jobs=int(self._simulation.get_active_job_count()),
                current_utilization=float(
                    self._simulation.get_current_utilization()
                ),
            )
        except _DR_EVT_ERRORS as error:
            raise InfrastructureFailure(
                f"cannot snapshot platform {self.name!r}: {error}"
            ) from error

    def timings(self, handles: Sequence[int]) -> list[JobTiming]:
        """Return adapter timing values, preserving requested handle order."""
        self._ensure_open()
        requested = list(handles)
        for handle in requested:
            if handle not in self._ledger:
                raise KeyError(
                    f"platform {self.name!r} has no job handle {handle}"
                )
        try:
            raw_timings = self._simulation.get_job_timings(requested)
        except _DR_EVT_ERRORS as error:
            raise InfrastructureFailure(
                f"cannot read timings from platform {self.name!r}: {error}"
            ) from error

        return [
            JobTiming(
                handle=int(timing.job_idx),
                key=self._ledger[int(timing.job_idx)],
                submit_s=float(timing.submit_time),
                begin_s=float(timing.begin_time),
                end_s=float(timing.end_time),
                limit_s=int(timing.limit_time),
                actual_run_s=float(timing.actual_run_time),
                num_nodes=int(timing.num_nodes),
                scheduled=bool(timing.scheduled),
            )
            for timing in raw_timings
        ]

    def finish(self) -> PlatformReport:
        """Drain all work, write both traces, and return final records."""
        if self._report is not None:
            return self._report

        self.advance_to(max(self.now(), _DRAIN_TIME_S))
        timings = tuple(self.timings(self._handles))
        try:
            self._simulation.write_simulated_trace()
            self._simulation.write_resource_trace(
                str(self._resource_trace_path)
            )
            raw_statistics = self._simulation.get_statistics()
        except _DR_EVT_ERRORS as error:
            raise InfrastructureFailure(
                f"cannot finish platform {self.name!r}: {error}"
            ) from error

        statistics = {
            field: float(getattr(raw_statistics, field))
            for field in _STATISTIC_FIELDS
        }
        self._report = PlatformReport(
            name=self.name,
            timings=timings,
            statistics=statistics,
            simulated_trace_path=str(self._simulated_trace_path),
            resource_trace_path=str(self._resource_trace_path),
        )
        return self._report
