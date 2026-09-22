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
    source: str = ""
    user: str = ""
    persona: str = ""

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
                row.get("source") or "",
                row.get("user") or "",
                row.get("persona") or "",
            )
            if job.job_id in seen:
                raise ValueError(f"duplicate job_id {job.job_id!r}")
            seen.add(job.job_id)
            jobs.append(job)
    return jobs


def write_jobs(jobs: list[Job], path: str | Path) -> None:
    """Write jobs with stable columns and Unix line endings."""
    platforms = sorted(
        {name for job in jobs if isinstance(job.bid, dict) for name in job.bid}
    )
    fields = _JOB_FIELDS[:5] + tuple(f"bid:{name}" for name in platforms)
    fields += _JOB_FIELDS[5:]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for job in jobs:
            output = {
                "job_id": job.job_id,
                "job_submit_time": job.submit_s,
                "num_nodes": job.num_nodes,
                "time_limit": job.limit_s,
                "bid": "" if isinstance(job.bid, dict) else job.bid,
                "requires": " ".join(sorted(job.requires)),
                "runtime": "" if job.runtime_s is None else job.runtime_s,
                "source": job.source,
                "user": job.user,
                "persona": job.persona,
            }
            if isinstance(job.bid, dict):
                output.update(
                    {f"bid:{name}": job.bid.get(name, "") for name in platforms}
                )
            writer.writerow(output)
