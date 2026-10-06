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

Row = namedtuple("Row", "source order identity submit nodes limit runtime user ran")


def _text(row, field: str) -> str | None:
    """Return a field's text, or None when it is missing, blank or '-'."""
    value = (row.get(field) or "").strip()
    return None if value in {"", "-"} else value


def _count(row, field: str) -> int | None:
    """Return a whole-number field, or None when it is missing, blank or '-'."""
    value = _text(row, field)
    return None if value is None else math.floor(float(value))


def _identity(row, field: str, order: int) -> str:
    """Return a stable identity from a trace identifier or file order."""
    value = _text(row, field)
    return f"row:{order}" if value is None else f"job:{value}"


def read_lc(source: str, path: str | Path) -> list[Row]:
    """Read one pseudonymized LC trace."""
    records = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for order, row in enumerate(csv.DictReader(stream)):
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
                    _identity(row, "job.id", order),
                    math.floor(float(row["t_submit"])),
                    _count(row, "job.node.count"),
                    math.floor(limit),
                    runtime,
                    _text(row, "user.name"),
                    ran,
                )
            )
    return records


def read_simple(source: str, path: str | Path) -> list[Row]:
    """Read one DR_EVT simple trace."""
    records = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for order, row in enumerate(csv.DictReader(stream)):
            records.append(
                Row(
                    source,
                    order,
                    _identity(row, "job_id", order),
                    math.floor(float(row["job_submit_time"])),
                    _count(row, "num_nodes"),
                    math.floor(float(row["time_limit"])),
                    _count(row, "actual_run_time"),
                    _text(row, "user"),
                    True,
                )
            )
    return records
