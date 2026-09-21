################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Read the platform file and the job stream.

``platforms.csv`` follows dr_evt's ``sync_systems.csv``: ``system_id``,
``total_nodes``, ``price_per_node_hour``, an optional ``hardware`` column of
space-separated tags, and an optional ``address`` for a gRPC server.

``jobs.csv`` is a dr_evt trace, ``job_submit_time``, ``num_nodes``,
``time_limit``, with a ``job_id``, an optional ``leg_id`` (one row per leg, in
order), an optional ``requires`` column of hardware tags, and the bid: either
one ``bid`` column holding the job's multiplier on its base cost, or one
``bid:<platform>`` column per platform holding a multiplier on that platform's
cost, blank meaning the platform is not bid on.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re

from .mechanisms import Job, Leg, Platform

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    source = Path(path)
    try:
        with source.open(newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames is None:
                raise ValueError(f"{source}: missing CSV header")
            fields = [field.strip() for field in reader.fieldnames]
            if len(set(fields)) != len(fields):
                raise ValueError(f"{source}: duplicate CSV columns")
            return fields, list(reader)
    except OSError as error:
        raise ValueError(f"cannot read {source}: {error}") from error


def _require(path: str | Path, fields: list[str], names: set[str]) -> None:
    missing = sorted(names - set(fields))
    if missing:
        raise ValueError(f"{path}: row 1 is missing columns: {', '.join(missing)}")


def _int(row: dict[str, str], name: str, row_number: int) -> int:
    try:
        return int((row.get(name) or "").strip())
    except ValueError as error:
        raise ValueError(f"row {row_number}: {name} must be an integer") from error


def _float(text: str, name: str, row_number: int) -> float:
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"row {row_number}: {name} must be a number") from error


def _tags(text: str | None) -> frozenset[str]:
    return frozenset((text or "").split())


def read_platforms(path: str | Path) -> list[Platform]:
    """Read the platforms, in file order."""
    fields, rows = _rows(path)
    _require(path, fields, {"system_id", "total_nodes", "price_per_node_hour"})
    platforms = []
    seen = set()
    for row_number, row in enumerate(rows, start=2):
        name = (row.get("system_id") or "").strip()
        if not _SAFE_NAME.fullmatch(name) or name in {".", ".."}:
            raise ValueError(f"row {row_number}: system_id must be filename safe")
        if name in seen:
            raise ValueError(f"row {row_number}: duplicate system_id {name!r}")
        seen.add(name)
        platforms.append(
            Platform(
                name,
                _int(row, "total_nodes", row_number),
                _float(
                    (row.get("price_per_node_hour") or "").strip(),
                    "price_per_node_hour",
                    row_number,
                ),
                _tags(row.get("hardware")),
                (row.get("address") or "").strip() or None,
            )
        )
    return platforms


def read_jobs(path: str | Path) -> list[Job]:
    """Read the job stream, in file order, legs grouped by job id."""
    fields, rows = _rows(path)
    _require(path, fields, {"job_id", "job_submit_time", "num_nodes", "time_limit"})
    platform_columns = [field for field in fields if field.startswith("bid:")]
    if ("bid" in fields) == bool(platform_columns):
        raise ValueError(
            f"{path}: give either one bid column or bid:<platform> columns"
        )

    order: list[str] = []
    submits: dict[str, int] = {}
    legs: dict[str, list[Leg]] = {}
    bids: dict[str, float | dict[str, float]] = {}
    for row_number, row in enumerate(rows, start=2):
        job_id = (row.get("job_id") or "").strip()
        if not job_id:
            raise ValueError(f"row {row_number}: job_id must not be blank")
        submit_s = _int(row, "job_submit_time", row_number)
        leg = Leg(
            (row.get("leg_id") or "0").strip() or "0",
            _int(row, "num_nodes", row_number),
            _int(row, "time_limit", row_number),
            _tags(row.get("requires")),
        )
        if platform_columns:
            bid: float | dict[str, float] = {
                column[len("bid:") :]: _float(text, column, row_number)
                for column in platform_columns
                if (text := (row.get(column) or "").strip())
            }
            if not bid:
                raise ValueError(f"row {row_number}: no platform is bid on")
        else:
            bid = _float((row.get("bid") or "").strip(), "bid", row_number)

        if job_id not in submits:
            order.append(job_id)
            submits[job_id] = submit_s
            legs[job_id] = []
            bids[job_id] = bid
        elif submits[job_id] != submit_s:
            raise ValueError(f"row {row_number}: job {job_id!r} submit times differ")
        elif bids[job_id] != bid:
            raise ValueError(f"row {row_number}: job {job_id!r} bids differ by leg")
        if any(existing.leg_id == leg.leg_id for existing in legs[job_id]):
            raise ValueError(f"row {row_number}: duplicate leg_id {leg.leg_id!r}")
        legs[job_id].append(leg)

    return [
        Job(job_id, submits[job_id], tuple(legs[job_id]), bids[job_id])
        for job_id in order
    ]


__all__ = ["read_jobs", "read_platforms"]
