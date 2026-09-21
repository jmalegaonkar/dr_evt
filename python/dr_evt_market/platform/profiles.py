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
    """The Corona AMD GPU platform profile."""

    name = "corona"
    total_nodes = 121
    price_per_node_hour = 2.0
    hardware = frozenset({"cpu", "gpu", "amd"})


class Dane(Platform):
    """The Dane CPU platform profile."""

    name = "dane"
    total_nodes = 1544
    price_per_node_hour = 1.0
    hardware = frozenset({"cpu"})


class Lassen(Platform):
    """The Lassen NVIDIA GPU platform profile."""

    name = "lassen"
    total_nodes = 795
    price_per_node_hour = 3.0
    hardware = frozenset({"cpu", "gpu", "nvidia"})


class Tioga(Platform):
    """The Tioga AMD GPU platform profile."""

    name = "tioga"
    total_nodes = 32
    price_per_node_hour = 6.0
    hardware = frozenset({"cpu", "gpu", "amd"})


class Tuolumne(Platform):
    """The Tuolumne AMD GPU platform profile."""

    name = "tuolumne"
    total_nodes = 1152
    price_per_node_hour = 8.0
    hardware = frozenset({"cpu", "gpu", "amd"})


PLATFORMS = (Corona, Dane, Lassen, Tioga, Tuolumne)
DEFAULT_FEDERATION = ("corona", "lassen", "tioga", "tuolumne")
_PROFILES = {profile.name: profile for profile in PLATFORMS}


def federation(work_dir, share=1.0, names=DEFAULT_FEDERATION) -> dict[str, Platform]:
    """Build the selected named platforms in the requested order."""
    root = Path(work_dir)
    try:
        return {name: _PROFILES[name](root / name, share) for name in names}
    except KeyError as error:
        raise ValueError(f"unknown platform {error.args[0]!r}") from None
