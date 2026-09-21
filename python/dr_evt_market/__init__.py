################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""A market over DR_EVT clusters: the platform contract, the auction, the loop."""

from .controller import (
    Controller,
    RoutedLeg,
    RoutingError,
    RunReport,
    WindowRecord,
    write_outputs,
)
from .inputs import read_jobs, read_platforms
from .mechanisms import (
    Decision,
    Job,
    Leg,
    Mechanism,
    Placement,
    Platform,
    Rejection,
    Vcg,
    Window,
    base_cost,
    build_window,
    demand,
    placements,
    submit,
    validate_decisions,
)
from .platforms.base import (
    ClockViolation,
    ConfigurationError,
    InfrastructureFailure,
    JobTiming,
    PlatformReport,
    PlatformSession,
    PlatformSnapshot,
    StructuralRejection,
    SubmitRequest,
    validate,
)
from .platforms.grpc import GrpcPlatform, SessionClient
from .platforms.inprocess import InProcessPlatform
from .platforms.server import ServerProcess

__version__ = "0.1.0"

__all__ = [
    "ClockViolation",
    "ConfigurationError",
    "Controller",
    "Decision",
    "GrpcPlatform",
    "InProcessPlatform",
    "InfrastructureFailure",
    "Job",
    "JobTiming",
    "Leg",
    "Mechanism",
    "Placement",
    "Platform",
    "PlatformReport",
    "PlatformSession",
    "PlatformSnapshot",
    "Rejection",
    "RoutedLeg",
    "RoutingError",
    "RunReport",
    "ServerProcess",
    "SessionClient",
    "StructuralRejection",
    "SubmitRequest",
    "Vcg",
    "Window",
    "WindowRecord",
    "base_cost",
    "build_window",
    "demand",
    "placements",
    "read_jobs",
    "read_platforms",
    "submit",
    "validate",
    "validate_decisions",
    "write_outputs",
]
