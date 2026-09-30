################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Mechanisms: the interface, candidates and offers, and the auctions."""

from .base import Decision, Mechanism, candidates, offers
from .firstfit import FirstFit
from .firstprice import FirstPrice
from .regret import grid_regret, refined_regret
from .regretformer import RegretFormer
from .training import record_windows, train_regretformer
from .vcg import Vcg

MECHANISMS = {
    "vcg": Vcg,
    "firstprice": FirstPrice,
    "firstfit": FirstFit,
    "regretformer": RegretFormer,
}

__all__ = [
    "Decision",
    "FirstFit",
    "FirstPrice",
    "MECHANISMS",
    "Mechanism",
    "RegretFormer",
    "Vcg",
    "candidates",
    "grid_regret",
    "offers",
    "record_windows",
    "refined_regret",
    "train_regretformer",
]
