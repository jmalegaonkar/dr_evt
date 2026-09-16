################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Read platform and bid-bearing job CSV files for market runs."""

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import re

from .mechanisms import JobBid, LegBid, LegSpec, QueuedJob

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class PlatformSpec:
    """Describe one priced platform and its optional gRPC address."""

    system_id: str
    total_nodes: int
    price_per_node_hour: float
    address: str | None = None


def _rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    source = Path(path)
    try:
        with source.open(newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames is None:
                raise ValueError(f"{source}: missing CSV header")
            fieldnames = [field.strip() for field in reader.fieldnames]
            if len(set(fieldnames)) != len(fieldnames):
                raise ValueError(f"{source}: duplicate CSV columns")
            return fieldnames, list(reader)
    except OSError as error:
        raise ValueError(f"cannot read {source}: {error}") from error


def _required_columns(
    path: str | Path,
    fieldnames: list[str],
    required: set[str],
) -> None:
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(
            f"{path}: row 1 is missing columns: {', '.join(missing)}"
        )


def _integer(value: str, field_name: str, row_number: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"row {row_number}: {field_name} must be an integer"
        ) from error
    return result


def _nonnegative_float(value: str, field_name: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"row {row_number}: {field_name} must be a number"
        ) from error
    if not isfinite(result) or result < 0.0:
        raise ValueError(
            f"row {row_number}: {field_name} must be finite and non-negative"
        )
    return result


def read_platforms(path: str | Path) -> list[PlatformSpec]:
    """Read unique, filename-safe platform definitions from CSV."""
    fieldnames, rows = _rows(path)
    _required_columns(
        path,
        fieldnames,
        {"system_id", "total_nodes", "price_per_node_hour"},
    )
    platforms = []
    names = set()
    for row_number, row in enumerate(rows, start=2):
        name = (row.get("system_id") or "").strip()
        if not _SAFE_NAME.fullmatch(name) or name in {".", ".."}:
            raise ValueError(
                f"row {row_number}: system_id must be filename safe"
            )
        if name in names:
            raise ValueError(f"row {row_number}: duplicate system_id {name!r}")
        total_nodes = _integer(
            row.get("total_nodes") or "",
            "total_nodes",
            row_number,
        )
        if total_nodes < 1:
            raise ValueError(
                f"row {row_number}: total_nodes must be positive"
            )
        price = _nonnegative_float(
            row.get("price_per_node_hour") or "",
            "price_per_node_hour",
            row_number,
        )
        address = (row.get("address") or "").strip() or None
        platforms.append(PlatformSpec(name, total_nodes, price, address))
        names.add(name)
    return platforms


def read_jobs(
    path: str | Path,
    platforms: list[PlatformSpec],
) -> tuple[list[QueuedJob], dict[str, JobBid]]:
    """Read ordered jobs, composite legs, and per-platform private bids."""
    fieldnames, rows = _rows(path)
    _required_columns(
        path,
        fieldnames,
        {"job_id", "job_submit_time", "num_nodes", "time_limit"},
    )
    platform_names = [platform.system_id for platform in platforms]
    if len(set(platform_names)) != len(platform_names):
        raise ValueError("platform system IDs must be unique")
    expected_bid_columns = {f"bid:{name}" for name in platform_names}
    actual_bid_columns = {
        field for field in fieldnames if field.startswith("bid:")
    }
    unknown = sorted(actual_bid_columns - expected_bid_columns)
    if unknown:
        raise ValueError(
            f"{path}: row 1 has unknown bid columns: {', '.join(unknown)}"
        )
    missing = sorted(expected_bid_columns - actual_bid_columns)
    if missing:
        raise ValueError(
            f"{path}: row 1 is missing bid columns: {', '.join(missing)}"
        )

    order = []
    submit_times: dict[str, int] = {}
    legs_by_job: dict[str, list[LegSpec]] = {}
    bids_by_job: dict[str, list[LegBid]] = {}
    leg_ids_by_job: dict[str, set[str]] = {}
    for row_number, row in enumerate(rows, start=2):
        job_id = (row.get("job_id") or "").strip()
        if not job_id:
            raise ValueError(f"row {row_number}: job_id must not be blank")
        submit_s = _integer(
            row.get("job_submit_time") or "",
            "job_submit_time",
            row_number,
        )
        if submit_s < 0:
            raise ValueError(
                f"row {row_number}: job_submit_time must be non-negative"
            )
        num_nodes = _integer(
            row.get("num_nodes") or "",
            "num_nodes",
            row_number,
        )
        limit_s = _integer(
            row.get("time_limit") or "",
            "time_limit",
            row_number,
        )
        if num_nodes < 1 or limit_s < 1:
            raise ValueError(
                f"row {row_number}: num_nodes and time_limit must be positive"
            )
        leg_id = (row.get("leg_id") or "0").strip() or "0"
        values = {}
        for platform_name in platform_names:
            field_name = f"bid:{platform_name}"
            raw_value = (row.get(field_name) or "").strip()
            if raw_value:
                values[platform_name] = _nonnegative_float(
                    raw_value,
                    field_name,
                    row_number,
                )
        if not values:
            raise ValueError(
                f"row {row_number}: leg has no acceptable platform"
            )

        if job_id not in submit_times:
            order.append(job_id)
            submit_times[job_id] = submit_s
            legs_by_job[job_id] = []
            bids_by_job[job_id] = []
            leg_ids_by_job[job_id] = set()
        elif submit_times[job_id] != submit_s:
            raise ValueError(
                f"row {row_number}: job {job_id!r} has inconsistent submit time"
            )
        if leg_id in leg_ids_by_job[job_id]:
            raise ValueError(
                f"row {row_number}: duplicate leg_id {leg_id!r} for {job_id!r}"
            )
        leg_ids_by_job[job_id].add(leg_id)
        legs_by_job[job_id].append(LegSpec(leg_id, num_nodes, limit_s))
        bids_by_job[job_id].append(LegBid(leg_id, values))

    jobs = [
        QueuedJob(job_id, submit_times[job_id], tuple(legs_by_job[job_id]))
        for job_id in order
    ]
    bids = {
        job_id: JobBid(job_id, tuple(bids_by_job[job_id]))
        for job_id in order
    }
    return jobs, bids
