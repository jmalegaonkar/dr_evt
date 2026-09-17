################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Load learned-mechanism windows logged by market controller runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..mechanisms.base import (
    JobBid,
    JobOffer,
    LegBid,
    LegSpec,
    MarketObservation,
    Placement,
)
from .windows import WindowStructure, structure_from_observation


def _observation(payload: dict) -> MarketObservation:
    offers = []
    bids = {}
    for job in payload["jobs"]:
        legs = tuple(
            LegSpec(
                str(leg["leg_id"]),
                int(leg["num_nodes"]),
                int(leg["limit_s"]),
            )
            for leg in job["legs"]
        )
        candidates = tuple(
            Placement(
                str(candidate["placement_id"]),
                candidate["platform_by_leg"],
                {
                    name: int(nodes)
                    for name, nodes in candidate["demand_by_platform"].items()
                },
                float(candidate["resource_cost_credits"]),
            )
            for candidate in job["candidates"]
        )
        job_id = str(job["job_id"])
        offers.append(JobOffer(
            job_id,
            int(job["submit_s"]),
            legs,
            candidates,
        ))
        bids[job_id] = JobBid(
            job_id,
            tuple(
                LegBid(
                    str(leg["leg_id"]),
                    {
                        name: float(value)
                        for name, value in leg["value_by_platform"].items()
                    },
                )
                for leg in job["legs"]
            ),
        )
    return MarketObservation(
        int(payload["time_s"]),
        int(payload["window_index"]),
        int(payload["seed"]),
        tuple(offers),
        bids,
        {
            name: int(nodes)
            for name, nodes in payload["free_nodes"].items()
        },
        tuple(str(job_id) for job_id in payload.get("truncated_jobs", ())),
    )


def harvest_structures(directory: str | Path) -> list[WindowStructure]:
    """Read ``windows.csv`` and its logged JSON observations in index order."""
    root = Path(directory)
    windows_path = root / "windows.csv"
    try:
        with windows_path.open(newline="", encoding="utf-8") as input_file:
            rows = list(csv.DictReader(input_file))
    except OSError as error:
        raise ValueError(f"cannot read {windows_path}: {error}") from error
    indexes = [int(row["index"]) for row in rows]
    structures = []
    for index in indexes:
        path = root / f"observation_{index:06d}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read {path}: {error}") from error
        if int(payload.get("window_index", -1)) != index:
            raise ValueError(f"{path}: window index does not match windows.csv")
        observation = _observation(payload)
        if observation.jobs:
            structures.append(structure_from_observation(observation))
    if not structures:
        raise ValueError(f"{root}: no non-empty logged observations")
    return structures
