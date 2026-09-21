################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The job record and the jobs file."""

import csv
from dataclasses import dataclass
from pathlib import Path

_JOB_FIELDS = (
    "job_id",
    "job_submit_time",
    "num_nodes",
    "time_limit",
    "bid",
    "requires",
    "runtime",
    "source",
    "user",
    "persona",
)


@dataclass(frozen=True)
class Job:
    """One job offered to the federation market."""

    job_id: str
    submit_s: int
    num_nodes: int
    limit_s: int
    bid: float | dict[str, float]
    requires: frozenset[str] = frozenset()
    runtime_s: int | None = None

    def __post_init__(self) -> None:
        """Freeze the hardware requirements."""
        object.__setattr__(self, "requires", frozenset(self.requires))
        if isinstance(self.bid, dict):
            object.__setattr__(self, "bid", dict(sorted(self.bid.items())))

    def multiplier(self, platform: str) -> float | None:
        """Return the multiplier offered on a platform, if any."""
        return self.bid.get(platform) if isinstance(self.bid, dict) else self.bid


def read_jobs(path: str | Path) -> list[Job]:
    """Read market jobs from a CSV file in row order."""
    jobs, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or ()
        bid_fields = [name for name in fields if name.startswith("bid:")]
        for row in reader:
            runtime = row.get("runtime")
            platform_bid = {
                name[4:]: float(row[name]) for name in bid_fields if row[name] != ""
            }
            job = Job(
                row["job_id"],
                int(row["job_submit_time"]),
                int(row["num_nodes"]),
                int(row["time_limit"]),
                platform_bid or float(row["bid"]),
                frozenset((row.get("requires") or "").split()),
                None if runtime in (None, "") else int(runtime),
            )
            if job.job_id in seen:
                raise ValueError(f"duplicate job_id {job.job_id!r}")
            seen.add(job.job_id)
            jobs.append(job)
    return jobs


def write_jobs(rows: list[dict], path: str | Path) -> None:
    """Write prepared job rows with stable columns and Unix line endings."""
    platforms = sorted(
        {name for row in rows if isinstance(row["bid"], dict) for name in row["bid"]}
    )
    fields = _JOB_FIELDS[:5] + tuple(f"bid:{name}" for name in platforms)
    fields += _JOB_FIELDS[5:]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            if isinstance(output["bid"], dict):
                bid = output["bid"]
                output["bid"] = ""
                output.update({f"bid:{name}": bid.get(name, "") for name in platforms})
            writer.writerow(output)
