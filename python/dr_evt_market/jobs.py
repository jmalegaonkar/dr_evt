################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""The job stream: one row per job, and the trace preparation with persona bids."""

import csv
import hashlib
import math
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


def _read_trace(source: str, path: str | Path, trace_format: str) -> list[dict]:
    records = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for order, row in enumerate(csv.DictReader(stream)):
            if trace_format == "lc":
                raw_nodes = row["job.node.count"]
                nodes = None if raw_nodes in {"", "-"} else int(float(raw_nodes))
                no_start = row["t_run"].strip() == "-"
                if no_start:
                    runtime = None
                else:
                    begin = math.floor(float(row["t_run"]))
                    end = math.floor(float(row["t_inactive"]))
                    runtime = end - begin
                try:
                    limit = float(row["user_time_limit"])
                except (TypeError, ValueError):
                    limit = float(row["time_limit"])
                submit = math.floor(float(row["t_submit"]))
                user = row["user.name"]
            else:
                raw_nodes = row["num_nodes"]
                nodes = None if raw_nodes in {"", "-"} else int(float(raw_nodes))
                limit = float(row["time_limit"])
                raw_runtime = row.get("actual_run_time")
                runtime = (
                    None
                    if raw_runtime in (None, "")
                    else math.floor(float(raw_runtime))
                )
                no_start = False
                submit = math.floor(float(row["job_submit_time"]))
                user = row.get("user")
            records.append(
                dict(
                    source=source,
                    order=order,
                    submit=submit,
                    nodes=nodes,
                    limit=math.floor(limit),
                    runtime=runtime,
                    user=user,
                    no_start=no_start,
                )
            )
    return records


def _persona_bid(
    seed: int, source: str, user: str, job_id: str, per_platform
) -> tuple[str, float | dict[str, float]]:
    import numpy

    def generator(key: str):
        digest = hashlib.sha256(key.encode()).digest()[:8]
        return numpy.random.default_rng(int.from_bytes(digest, "big"))

    user_rng = generator(f"{seed}:{source}:{user}")
    job_rng = generator(f"{seed}:{source}:{user}:{job_id}")
    draw = user_rng.random()
    heavy = user_rng.random() < 0.2
    if draw < 0.45:
        persona, multiplier = "sticker", 1.0
    elif draw < 0.80:
        urgent = job_rng.random() < 0.2
        persona = "tier"
        multiplier = 4.0 if urgent and heavy else 2.0 if urgent else 1.0
    elif draw < 0.95:
        value = job_rng.lognormal(math.log(3.0), 0.5)
        persona, multiplier = "value", round(value, 4)
    else:
        persona, multiplier = "whale", 10.0
    if per_platform is None:
        return persona, multiplier
    bid = {
        name: round(multiplier * math.exp(user_rng.normal(0.0, 0.3)), 4)
        for name in per_platform
    }
    return persona, bid


def prepare(
    traces,
    *,
    trace_format="lc",
    start=None,
    hours=None,
    seed=0,
    requires="gpu",
    per_platform=None,
) -> tuple[list[Job], dict[str, int], list[dict]]:
    """Prepare deterministic jobs from one interval across named traces."""
    if trace_format not in {"lc", "simple"}:
        raise ValueError("trace_format must be 'lc' or 'simple'")
    records = []
    for source, path in traces.items():
        records.extend(_read_trace(source, path, trace_format))
    records.sort(key=lambda row: (row["submit"], row["source"], row["order"]))

    lower = (
        records[0]["submit"]
        if start is None and records
        else math.floor(float(start or 0))
    )
    upper = math.inf if hours is None else lower + float(hours) * 3600
    selected = [row for row in records if lower <= row["submit"] < upper]
    summary = {
        "read": len(records),
        "kept": 0,
        "no_nodes": 0,
        "no_limit": 0,
        "no_start": 0,
        "bad_runtime": 0,
        "outside_interval": len(records) - len(selected),
    }
    summary.update({f"kept:{source}": 0 for source in traces})

    kept = []
    for record in selected:
        if record["nodes"] is None or record["nodes"] < 1:
            reason = "no_nodes"
        elif record["limit"] < 1:
            reason = "no_limit"
        elif record["no_start"]:
            reason = "no_start"
        elif record["runtime"] is not None and record["runtime"] < 1:
            reason = "bad_runtime"
        else:
            kept.append(record)
            continue
        summary[reason] += 1

    origin = kept[0]["submit"] if kept else 0
    requirement_text = " ".join(requires.split())
    requirement_set = frozenset(requirement_text.split())
    jobs, rows = [], []
    for index, record in enumerate(kept, start=1):
        job_id = f"j{index:06d}"
        user = record["user"] if record["user"] is not None else job_id
        persona, bid = _persona_bid(seed, record["source"], user, job_id, per_platform)
        job = Job(
            job_id,
            record["submit"] - origin,
            record["nodes"],
            record["limit"],
            bid,
            requirement_set,
            record["runtime"],
        )
        jobs.append(job)
        rows.append(
            dict(
                zip(
                    _JOB_FIELDS,
                    (
                        job.job_id,
                        job.submit_s,
                        job.num_nodes,
                        job.limit_s,
                        job.bid,
                        requirement_text,
                        "" if job.runtime_s is None else job.runtime_s,
                        record["source"],
                        user,
                        persona,
                    ),
                )
            )
        )
        summary[f"kept:{record['source']}"] += 1
    summary["kept"] = len(jobs)
    return jobs, summary, rows


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
