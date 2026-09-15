################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Pure Python contract shared by DR_EVT platform adapters."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SubmitRequest:
    """Describe one client-owned job or composite leg to submit."""

    key: str
    submit_s: int
    num_nodes: int
    limit_s: int
    q_id: str = "1"


@dataclass(frozen=True)
class JobTiming:
    """Mirror one DR_EVT job timing record in seconds."""

    handle: int
    key: str
    submit_s: float
    begin_s: float
    end_s: float
    limit_s: int
    actual_run_s: float
    num_nodes: int
    scheduled: bool


@dataclass(frozen=True)
class ResourceRelease:
    """Describe nodes projected to become available at one time."""

    time_s: float
    nodes_released: int


@dataclass(frozen=True)
class PlatformSnapshot:
    """Capture scheduler capacity and reservation state at one time."""

    name: str
    time_s: int
    total_nodes: int
    free_nodes: int
    in_use_nodes: int
    waiting_jobs: int
    shadow_time_s: float
    releases: tuple[ResourceRelease, ...]
    current_utilization: float | None = None
    resource_area: float | None = None
    prediction_horizon_s: float | None = None


@dataclass(frozen=True)
class PlatformReport:
    """Collect final timing records, statistics, and output paths."""

    name: str
    timings: tuple[JobTiming, ...]
    statistics: dict[str, float]
    simulated_trace_path: str | None
    resource_trace_path: str | None


class ClockViolation(ValueError):
    """Report a non-integer, backward, or out-of-order time."""


class ConfigurationError(ValueError):
    """Report an invalid adapter or scheduler configuration."""


class StructuralRejection(ValueError):
    """Report a request that cannot fit any valid scheduler state."""


class InfrastructureFailure(RuntimeError):
    """Report a transport, stream, or server process failure."""


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate(request: SubmitRequest) -> None:
    """Validate the context-independent shape of one submission request."""

    if not _is_integer(request.submit_s):
        raise ClockViolation("submit_s must be an integer number of seconds")
    if not _is_integer(request.num_nodes) or request.num_nodes < 1:
        raise StructuralRejection("num_nodes must be a positive integer")
    if not _is_integer(request.limit_s) or request.limit_s < 1:
        raise StructuralRejection("limit_s must be a positive integer")
    if (not isinstance(request.q_id, str) or not request.q_id.isdigit()
            or not 1 <= int(request.q_id) <= 10):
        raise ConfigurationError("q_id must be a digit string in 1..10")


class PlatformSession(Protocol):
    """Define the common lifecycle of one simulated platform session."""

    name: str
    total_nodes: int

    def now(self) -> int:
        """Return the adapter's current integer simulation time."""
        ...

    def submit(self, jobs: Sequence[SubmitRequest]) -> list[int]:
        """Validate and submit a batch, returning DR_EVT handles in order."""
        ...

    def advance_to(self, time_s: int) -> None:
        """Advance through all scheduler events at or before time_s."""
        ...

    def snapshot(self) -> PlatformSnapshot:
        """Return one consistent snapshot of current scheduler state."""
        ...

    def timings(self, handles: Sequence[int]) -> list[JobTiming]:
        """Return timing records for the requested handles in order."""
        ...

    def finish(self) -> PlatformReport:
        """Drain submitted work and return final records and output paths."""
        ...
