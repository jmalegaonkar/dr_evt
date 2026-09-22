################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Reading raw traces: the LC format and DR_EVT's simple format."""

import csv
import math
from collections import namedtuple
from pathlib import Path

Row = namedtuple("Row", "source order submit nodes limit runtime user ran")


def read_lc(source: str, path: str | Path) -> list[Row]:
    """Read one pseudonymized LC trace."""
    records = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for order, row in enumerate(csv.DictReader(stream)):
            raw_nodes = row["job.node.count"]
            nodes = None if raw_nodes in {"", "-"} else int(float(raw_nodes))
            ran = row["t_run"].strip() != "-"
            if ran:
                begin = math.floor(float(row["t_run"]))
                end = math.floor(float(row["t_inactive"]))
                runtime = end - begin
            else:
                runtime = None
            try:
                limit = float(row["user_time_limit"])
            except (TypeError, ValueError):
                limit = float(row["time_limit"])
            records.append(
                Row(
                    source,
                    order,
                    math.floor(float(row["t_submit"])),
                    nodes,
                    math.floor(limit),
                    runtime,
                    row["user.name"],
                    ran,
                )
            )
    return records


def read_simple(source: str, path: str | Path) -> list[Row]:
    """Read one DR_EVT simple trace."""
    records = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for order, row in enumerate(csv.DictReader(stream)):
            raw_nodes = row["num_nodes"]
            nodes = None if raw_nodes in {"", "-"} else int(float(raw_nodes))
            raw_runtime = row.get("actual_run_time")
            runtime = (
                None if raw_runtime in (None, "") else math.floor(float(raw_runtime))
            )
            records.append(
                Row(
                    source,
                    order,
                    math.floor(float(row["job_submit_time"])),
                    nodes,
                    math.floor(float(row["time_limit"])),
                    runtime,
                    row.get("user"),
                    True,
                )
            )
    return records
