################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The named machines, their federation, and how to build one."""

from pathlib import Path

from .base import Platform


class Corona(Platform):
    """The Corona GPU platform profile."""

    name = "corona"
    total_nodes = 121
    price_per_node_hour = 1.5
    hardware = frozenset({"cpu", "gpu"})


class Dane(Platform):
    """The Dane CPU platform profile."""

    name = "dane"
    total_nodes = 1544
    price_per_node_hour = 0.18
    hardware = frozenset({"cpu"})


class Matrix(Platform):
    """The Matrix GPU platform profile."""

    name = "matrix"
    total_nodes = 30
    price_per_node_hour = 1.6
    hardware = frozenset({"cpu", "gpu"})


class Tioga(Platform):
    """The Tioga GPU platform profile."""

    name = "tioga"
    total_nodes = 32
    price_per_node_hour = 2.7
    hardware = frozenset({"cpu", "gpu"})


class Tuolumne(Platform):
    """The Tuolumne GPU platform profile."""

    name = "tuolumne"
    total_nodes = 1152
    price_per_node_hour = 0.19
    hardware = frozenset({"cpu", "gpu"})


PLATFORMS = (Corona, Dane, Matrix, Tioga, Tuolumne)
DEFAULT_FEDERATION = ("corona", "dane", "matrix", "tioga", "tuolumne")
_PROFILES = {profile.name: profile for profile in PLATFORMS}


def federation(work_dir, share=1.0, names=DEFAULT_FEDERATION) -> dict[str, Platform]:
    """Build the selected named platforms in the requested order."""
    root = Path(work_dir)
    try:
        return {name: _PROFILES[name](root / name, share) for name in names}
    except KeyError as error:
        raise ValueError(f"unknown platform {error.args[0]!r}") from None
