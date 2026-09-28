################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Mechanisms: the interface, the candidates a job can take, and the auctions."""

from .base import Decision, Mechanism, candidates
from .firstprice import FirstPrice
from .regret import grid_regret, refined_regret
from .regretformer import RegretFormer
from .vcg import Vcg

MECHANISMS = {"vcg": Vcg, "firstprice": FirstPrice, "regretformer": RegretFormer}

__all__ = [
    "Decision",
    "FirstPrice",
    "MECHANISMS",
    "Mechanism",
    "RegretFormer",
    "Vcg",
    "candidates",
    "grid_regret",
    "refined_regret",
]
