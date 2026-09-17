################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Public types and interfaces for DR_EVT market mechanisms."""

from .base import (
    Decision,
    JobBid,
    JobOffer,
    LegBid,
    LegSpec,
    MarketObservation,
    Mechanism,
    Placement,
    RejectedDecision,
    validate_decisions,
)
from .clearing import QueuedJob, build_observation, submit_decisions
from .vcg import Vcg

__all__ = [
    "Decision",
    "JobBid",
    "JobOffer",
    "LegBid",
    "LegSpec",
    "MarketObservation",
    "Mechanism",
    "Placement",
    "QueuedJob",
    "RejectedDecision",
    "RegretFormer",
    "Vcg",
    "build_observation",
    "submit_decisions",
    "validate_decisions",
]


def __getattr__(name: str):
    """Load torch-dependent mechanism classes only when requested."""
    if name == "RegretFormer":
        from .regretformer import RegretFormer

        globals()[name] = RegretFormer
        return RegretFormer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
