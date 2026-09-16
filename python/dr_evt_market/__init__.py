################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Public contract for DR_EVT-backed market platform adapters."""

from .platforms.base import (
    ClockViolation,
    ConfigurationError,
    InfrastructureFailure,
    JobTiming,
    PlatformReport,
    PlatformSession,
    PlatformSnapshot,
    ResourceRelease,
    StructuralRejection,
    SubmitRequest,
    validate,
)
from .platforms.inprocess import InProcessPlatform
from .platforms.grpc import GrpcPlatform, SessionClient
from .platforms.prediction import Prediction, predict_start
from .platforms.server import ServerProcess

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ClockViolation",
    "ConfigurationError",
    "InfrastructureFailure",
    "GrpcPlatform",
    "InProcessPlatform",
    "JobTiming",
    "PlatformReport",
    "PlatformSession",
    "PlatformSnapshot",
    "Prediction",
    "ResourceRelease",
    "ServerProcess",
    "SessionClient",
    "StructuralRejection",
    "SubmitRequest",
    "predict_start",
    "validate",
]
