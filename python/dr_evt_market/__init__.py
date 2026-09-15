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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ClockViolation",
    "ConfigurationError",
    "InfrastructureFailure",
    "JobTiming",
    "PlatformReport",
    "PlatformSession",
    "PlatformSnapshot",
    "ResourceRelease",
    "StructuralRejection",
    "SubmitRequest",
    "validate",
]
