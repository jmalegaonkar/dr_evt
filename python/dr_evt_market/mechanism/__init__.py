################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Mechanisms: the interface, the candidates a job can take, and the auctions."""

from .base import Decision, Mechanism, candidates
from .firstprice import FirstPrice
from .vcg import Vcg

MECHANISMS = {"vcg": Vcg, "firstprice": FirstPrice}

__all__ = [
    "Decision",
    "FirstPrice",
    "MECHANISMS",
    "Mechanism",
    "Vcg",
    "candidates",
]
