################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""gRPC implementation of the market platform adapter contract."""

from collections.abc import Iterator, Sequence
import importlib
from pathlib import Path
import queue
import re
import sys
import tempfile
from typing import Any

from .base import (
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

_DRAIN_TIME_S = 1_000_000_000_000
_SAFE_SESSION_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_STATISTIC_FIELDS = (
    "jobs_submitted",
    "jobs_completed",
    "jobs_running",
    "jobs_waiting",
    "current_time",
    "total_nodes",
    "nodes_in_use",
    "nodes_available",
    "resource_area",
    "utilization",
    "avg_wait_time",
    "avg_turnaround_time",
    "makespan",
)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _proto_path() -> Path:
    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "proto"
        / "dr_evt_service.proto"
    )
    if not path.is_file():
        raise ConfigurationError(f"DR_EVT service definition not found: {path}")
    return path


def _mapped_server_error(
    request_kind: str,
    address: str,
    server_message: str,
) -> Exception:
    message = f"{address}: {server_message}"
    if request_kind == "get_job_timings" and "get_job_timing()" in message:
        return KeyError(message)
    if "submit_time" in server_message and (
        "sorted" in server_message or "current" in server_message
    ):
        return ClockViolation(message)
    if (
        server_message.startswith("Unknown ")
        or server_message.startswith("session_name must")
    ):
        return ConfigurationError(message)
    return InfrastructureFailure(message)


class SessionClient:
    """Correlate synchronous calls over one bidirectional Session stream."""

    def __init__(self, address: str) -> None:
        """Generate service stubs and open one stream to address."""
        if not isinstance(address, str) or not address:
            raise ConfigurationError("server address must be a non-empty string")
        try:
            import grpc
            import grpc_tools.protoc
        except ImportError as error:
            raise InfrastructureFailure(
                "Python gRPC dependencies are missing; install dr_evt_market[grpc]"
            ) from error

        proto = _proto_path()
        self._generated_dir = tempfile.TemporaryDirectory(
            prefix="dr_evt_market_grpc_"
        )
        result = grpc_tools.protoc.main([
            "grpc_tools.protoc",
            f"--proto_path={proto.parent}",
            f"--python_out={self._generated_dir.name}",
            f"--grpc_python_out={self._generated_dir.name}",
            str(proto),
        ])
        if result:
            self._generated_dir.cleanup()
            raise InfrastructureFailure(
                f"protoc failed while generating stubs from {proto}"
            )

        sys.path.insert(0, self._generated_dir.name)
        try:
            self.messages = importlib.import_module("dr_evt_service_pb2")
            service = importlib.import_module("dr_evt_service_pb2_grpc")
        finally:
            sys.path.remove(self._generated_dir.name)

        self.address = address
        self._grpc: Any = grpc
        self._outgoing: queue.Queue[Any | None] = queue.Queue()
        self._request_id = 1
        self._closed = False
        self._channel = grpc.insecure_channel(address)
        stub = service.SimulationServiceStub(self._channel)
        self._responses = stub.Session(self._request_iterator())

    def _request_iterator(self) -> Iterator[Any]:
        while True:
            request = self._outgoing.get()
            if request is None:
                return
            yield request

    def call(self, request: Any) -> Any:
        """Send one request and return its correlated server response."""
        if self._closed:
            raise InfrastructureFailure(
                f"{self.address}: session client is closed"
            )
        request.request_id = self._request_id
        self._request_id += 1
        self._outgoing.put(request)
        try:
            response = next(self._responses)
        except StopIteration as error:
            raise InfrastructureFailure(
                f"{self.address}: server closed the stream"
            ) from error
        except self._grpc.RpcError as error:
            raise InfrastructureFailure(
                f"{self.address}: gRPC request failed: {error.details()}"
            ) from error
        if response.request_id != request.request_id:
            raise InfrastructureFailure(
                f"{self.address}: response id {response.request_id} does not "
                f"match request id {request.request_id}"
            )
        if response.HasField("error"):
            request_kind = request.WhichOneof("request") or ""
            raise _mapped_server_error(
                request_kind,
                self.address,
                response.error.message,
            )
        return response

    def close(self) -> None:
        """Close the request iterator, channel, and generated-stub directory."""
        if self._closed:
            return
        self._closed = True
        self._outgoing.put(None)
        self._channel.close()
        self._generated_dir.cleanup()


class GrpcPlatform:
    """Drive one DR_EVT simulation through its gRPC streaming service."""

    def __init__(
        self,
        name: str,
        total_nodes: int,
        address: str,
        work_dir_on_server: str | Path,
        *,
        session_name: str,
    ) -> None:
        """Initialize an EASY/FCFS session using a server-readable header."""
        if not isinstance(name, str) or not name:
            raise ConfigurationError("platform name must be a non-empty string")
        if not _is_integer(total_nodes) or total_nodes < 1:
            raise ConfigurationError("total_nodes must be a positive integer")
        if (
            not isinstance(session_name, str)
            or not _SAFE_SESSION_NAME.fullmatch(session_name)
            or session_name in {".", ".."}
        ):
            raise ConfigurationError(
                "session_name must be 1-128 filename-safe characters"
            )

        self.name = name
        self.total_nodes = total_nodes
        self._work_dir = Path(work_dir_on_server).resolve()
        try:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            header_path = self._work_dir / f"{session_name}.header.csv"
            header_path.write_text(
                "job_submit_time,num_nodes,time_limit\n",
                encoding="utf-8",
            )
        except OSError as error:
            raise InfrastructureFailure(
                f"cannot prepare server work directory {self._work_dir}"
            ) from error

        self._client = SessionClient(address)
        self._messages = self._client.messages
        try:
            response = self._client.call(self._messages.ClientMessage(
                init=self._messages.InitRequest(
                    total_nodes=total_nodes,
                    trace_format="simple",
                    timestamp_format="epoch",
                    backfill_policy="easy",
                    priority_policy="fcfs",
                    run_time_mode="limit",
                    msec_output=True,
                    infile=str(header_path),
                    queue_impl="circular",
                    session_name=session_name,
                )
            ))
        except Exception:
            self._client.close()
            raise
        self._session_id = response.init.session_id
        self._ledger: dict[int, str] = {}
        self._handles: list[int] = []
        self._last_submit_s: int | None = None
        self._report: PlatformReport | None = None

    def now(self) -> int:
        """Return the server simulation's current integer time."""
        response = self._client.call(self._messages.ClientMessage(
            get_current_time=self._messages.GetCurrentTimeRequest()
        ))
        return int(response.get_current_time.current_time)

    def submit(self, jobs: Sequence[SubmitRequest]) -> list[int]:
        """Validate and append one chronologically ordered request batch."""
        requests = list(jobs)
        previous_submit = self._last_submit_s
        for request in requests:
            validate(request)
            if request.num_nodes > self.total_nodes:
                raise StructuralRejection(
                    f"job {request.key!r} requests {request.num_nodes} nodes, "
                    f"but platform {self.name!r} has {self.total_nodes}"
                )
            if (
                previous_submit is not None
                and request.submit_s < previous_submit
            ):
                raise ClockViolation(
                    f"job {request.key!r} submits at {request.submit_s}, "
                    f"before the previous submission at {previous_submit}"
                )
            previous_submit = request.submit_s

        if not requests:
            return []

        current_time = self.now()
        for request in requests:
            if request.submit_s < current_time:
                raise ClockViolation(
                    f"job {request.key!r} submits at {request.submit_s}, "
                    f"before platform time {current_time}"
                )

        batch = self._messages.AppendJobsRequest(requests=[
            self._messages.JobAppendData(
                submit_time=request.submit_s,
                num_nodes=request.num_nodes,
                queue=request.q_id,
                limit_time=request.limit_s,
            )
            for request in requests
        ])
        response = self._client.call(self._messages.ClientMessage(
            append_jobs=batch
        ))
        handles = [int(handle) for handle in response.append_jobs.job_idx]
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
        if not _is_integer(time_s):
            raise ClockViolation("advance time must be an integer number of seconds")
        current_time = self.now()
        if time_s < current_time:
            raise ClockViolation(
                f"cannot advance platform {self.name!r} from "
                f"{current_time} back to {time_s}"
            )
        self._client.call(self._messages.ClientMessage(
            advance_to=self._messages.AdvanceToRequest(target_time=time_s)
        ))

    def snapshot(self) -> PlatformSnapshot:
        """Return capacity, queue, and utilization state from the server."""
        statistics = self._client.call(self._messages.ClientMessage(
            get_statistics=self._messages.GetStatisticsRequest()
        )).get_statistics
        current_utilization = self._client.call(self._messages.ClientMessage(
            get_current_utilization=(
                self._messages.GetCurrentUtilizationRequest()
            )
        )).get_current_utilization.utilization
        return PlatformSnapshot(
            name=self.name,
            time_s=int(statistics.current_time),
            total_nodes=self.total_nodes,
            free_nodes=int(statistics.nodes_available),
            in_use_nodes=int(statistics.nodes_in_use),
            waiting_jobs=int(statistics.jobs_waiting),
            current_utilization=float(current_utilization),
        )

    def timings(self, handles: Sequence[int]) -> list[JobTiming]:
        """Return timing values for handles in their requested order."""
        requested = list(handles)
        response = self._client.call(self._messages.ClientMessage(
            get_job_timings=self._messages.GetJobTimingsRequest(
                job_idx=requested
            )
        ))
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
            for timing in response.get_job_timings.timings
        ]

    def finish(self) -> PlatformReport:
        """Drain all work, fetch timings, and request final server outputs."""
        if self._report is not None:
            return self._report

        self.advance_to(max(self.now(), _DRAIN_TIME_S))
        timings = tuple(self.timings(self._handles))
        response = self._client.call(self._messages.ClientMessage(
            finish_simulation=self._messages.FinishSimulationRequest()
        )).finish_simulation
        statistics = {
            field: float(getattr(response.statistics, field))
            for field in _STATISTIC_FIELDS
        }
        simulated_path = self._work_dir / response.simulated_trace_file
        resource_path = self._work_dir / response.resource_trace_file
        self._report = PlatformReport(
            name=self.name,
            timings=timings,
            statistics=statistics,
            simulated_trace_path=str(simulated_path.resolve()),
            resource_trace_path=str(resource_path.resolve()),
        )
        self._client.close()
        return self._report
