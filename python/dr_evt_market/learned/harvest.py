################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Load the windows a controller run logged with ``--log-windows``."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..mechanisms import Job, Leg, Platform, build_window
from .windows import WindowStructure, structure_from_window


def _job(payload: dict) -> Job:
    bid = payload["bid"]
    return Job(
        str(payload["job_id"]),
        int(payload["submit_s"]),
        tuple(
            Leg(
                str(leg["leg_id"]),
                int(leg["num_nodes"]),
                int(leg["limit_s"]),
                frozenset(leg.get("requires", ())),
            )
            for leg in payload["legs"]
        ),
        (
            {str(k): float(v) for k, v in bid.items()}
            if isinstance(bid, dict)
            else float(bid)
        ),
    )


def harvest_structures(directory: str | Path) -> list[WindowStructure]:
    """Read ``windows.csv`` and the logged window files in index order."""
    root = Path(directory)
    try:
        with (root / "windows.csv").open(newline="", encoding="utf-8") as input_file:
            indexes = [int(row["index"]) for row in csv.DictReader(input_file)]
    except OSError as error:
        raise ValueError(f"cannot read {root / 'windows.csv'}: {error}") from error
    structures = []
    for index in indexes:
        path = root / f"window_{index:06d}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read {path}: {error}") from error
        if int(payload.get("index", -1)) != index:
            raise ValueError(f"{path}: index does not match windows.csv")
        platforms = {
            name: Platform(
                name,
                int(spec["total_nodes"]),
                float(spec["price_per_node_hour"]),
                frozenset(spec.get("hardware", ())),
            )
            for name, spec in payload["platforms"].items()
        }
        jobs = [_job(item) for item in payload["jobs"]]
        if not jobs:
            continue
        window = build_window(
            int(payload["time_s"]),
            index,
            jobs,
            platforms,
            {name: int(nodes) for name, nodes in payload["free_nodes"].items()},
        )
        structures.append(structure_from_window(window))
    if not structures:
        raise ValueError(f"{root}: no non-empty logged windows")
    return structures


__all__ = ["harvest_structures"]
