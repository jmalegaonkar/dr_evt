################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The market's data model, clearing step and mechanisms."""

from .base import (
    Decision,
    Job,
    Leg,
    Mechanism,
    Placement,
    Platform,
    Rejection,
    Window,
    demand,
    validate_decisions,
)
from .clearing import base_cost, build_window, placements, submit
from .vcg import Vcg

__all__ = [
    "Decision",
    "Job",
    "Leg",
    "Mechanism",
    "Placement",
    "Platform",
    "RegretFormer",
    "Rejection",
    "Vcg",
    "Window",
    "base_cost",
    "build_window",
    "demand",
    "placements",
    "submit",
    "validate_decisions",
]


def __getattr__(name: str):
    """Import the learned mechanism only when it is asked for."""
    if name == "RegretFormer":
        from .regretformer import RegretFormer

        return RegretFormer
    raise AttributeError(name)
