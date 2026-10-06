################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The named machines, their federation, and how to build one."""

from collections.abc import Mapping
from pathlib import Path

from .base import Platform

# Speeds were derived on 2026-09-28 from ranks == 1 rows by removing the
# "(via quartz)" marks, taking each platform column's median relative run time,
# and using its reciprocal. The sample has 635 rows across 8 applications.
# Corona has no matrix row and uses 1.0. Dane's CPU values are bimodal, so its
# median-based speed is sensitive to the aggregation rule.
#
# Node counts: Corona's and Tuolumne's are LLNL's published numbers. Tioga's page
# lists 24 compute nodes; it keeps 32 until its size is confirmed. Posted prices per
# node-hour, like Dane's and Matrix's node counts, were set on 2026-09-28.


class Corona(Platform):
    """The Corona GPU platform profile."""

    name = "corona"
    total_nodes = 121
    price_per_node_hour = 1.5
    hardware = frozenset({"cpu", "gpu"})
    speed = {"cpu": 1.0, "gpu": 1.0}


class Dane(Platform):
    """The Dane CPU platform profile."""

    name = "dane"
    total_nodes = 1544
    price_per_node_hour = 0.18
    hardware = frozenset({"cpu"})
    speed = {"cpu": 0.861}


class Matrix(Platform):
    """The Matrix GPU platform profile."""

    name = "matrix"
    total_nodes = 30
    price_per_node_hour = 1.6
    hardware = frozenset({"cpu", "gpu"})
    speed = {"cpu": 2.574, "gpu": 3.695}


class Tioga(Platform):
    """The Tioga GPU platform profile."""

    name = "tioga"
    total_nodes = 32
    price_per_node_hour = 2.7
    hardware = frozenset({"cpu", "gpu"})
    speed = {"cpu": 1.594, "gpu": 7.042}


class Tuolumne(Platform):
    """The Tuolumne GPU platform profile."""

    name = "tuolumne"
    total_nodes = 1152
    price_per_node_hour = 0.19
    hardware = frozenset({"cpu", "gpu"})
    speed = {"cpu": 1.401, "gpu": 3.313}


PLATFORMS = (Corona, Dane, Matrix, Tioga, Tuolumne)
DEFAULT_FEDERATION = ("corona", "dane", "matrix", "tioga", "tuolumne")
PROFILES = {profile.name: profile for profile in PLATFORMS}


def federation(work_dir, share=1.0, names=DEFAULT_FEDERATION) -> dict[str, Platform]:
    """Build the selected named platforms in the requested order."""
    root = Path(work_dir)
    mapped = isinstance(share, Mapping)
    try:
        return {
            name: PROFILES[name](root / name, share.get(name, 1.0) if mapped else share)
            for name in names
        }
    except KeyError as error:
        raise ValueError(f"unknown platform {error.args[0]!r}") from None
