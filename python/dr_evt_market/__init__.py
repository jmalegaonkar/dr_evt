################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Public contract for DR_EVT-backed market platform adapters."""

from .controller import (
    Controller,
    RejectedJob,
    RoutedLeg,
    RoutingInvariantViolation,
    RunReport,
    WindowRecord,
    write_outputs,
)
from .inputs import PlatformSpec, read_jobs, read_platforms
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
from .platforms.inprocess import InProcessPlatform
from .platforms.grpc import GrpcPlatform, SessionClient
from .platforms.server import ServerProcess

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ClockViolation",
    "ConfigurationError",
    "Controller",
    "InfrastructureFailure",
    "GrpcPlatform",
    "InProcessPlatform",
    "JobTiming",
    "PlatformReport",
    "PlatformSession",
    "PlatformSnapshot",
    "PlatformSpec",
    "RejectedJob",
    "RoutedLeg",
    "RoutingInvariantViolation",
    "RunReport",
    "ServerProcess",
    "SessionClient",
    "StructuralRejection",
    "SubmitRequest",
    "WindowRecord",
    "read_jobs",
    "read_platforms",
    "validate",
    "write_outputs",
]
