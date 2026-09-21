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
from numbers import Real
from pathlib import Path

_JOB_FIELDS = ("job_id", "job_submit_time", "num_nodes", "time_limit", "bid",
               "requires", "runtime", "user", "persona")

def _valid_integer(value, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum

@dataclass(frozen=True)
class Job:
    """One job offered to the federation market."""

    job_id: str
    submit_s: int
    num_nodes: int
    limit_s: int
    bid: float
    requires: frozenset[str] = frozenset()
    runtime_s: int | None = None

    def __post_init__(self) -> None:
        """Validate the job fields and freeze its hardware requirements."""
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("column job_id must be a non-empty string")
        integer_fields = (("job_submit_time", self.submit_s, 0),
                          ("num_nodes", self.num_nodes, 1),
                          ("time_limit", self.limit_s, 1))
        for name, value, minimum in integer_fields:
            if not _valid_integer(value, minimum):
                raise ValueError(f"column {name} must be an integer >= {minimum}")
        if not (isinstance(self.bid, Real) and not isinstance(self.bid, bool)
                and math.isfinite(self.bid) and self.bid >= 0):
            raise ValueError("column bid must be a finite non-negative number")
        if self.runtime_s is not None and not _valid_integer(self.runtime_s, 1):
            raise ValueError("column runtime must be an integer >= 1 or blank")
        object.__setattr__(self, "requires", frozenset(self.requires))

def _parse(value: str, column: str, parser=float, hint: str = ""):
    try:
        result = parser(value)
        if parser is float and not math.isfinite(result):
            raise ValueError
    except (TypeError, ValueError) as error:
        kind = "an integer" if parser is int else "numeric"
        raise ValueError(f"column {column} must be {kind}{hint}") from error
    return result

def read_jobs(path: str | Path) -> list[Job]:
    """Read market jobs from a CSV file, preserving their row order."""
    jobs, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"job_id", "job_submit_time", "num_nodes", "time_limit", "bid"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required column {sorted(missing)[0]}")
        for number, row in enumerate(reader, start=2):
            runtime = row.get("runtime")
            try:
                job = Job(
                    row["job_id"],
                    _parse(row["job_submit_time"], "job_submit_time", int),
                    _parse(row["num_nodes"], "num_nodes", int),
                    _parse(row["time_limit"], "time_limit", int),
                    _parse(row["bid"], "bid"),
                    frozenset((row.get("requires") or "").split()),
                    None if runtime in (None, "") else _parse(runtime, "runtime", int),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"row {number}: {error}") from error
            if job.job_id in seen:
                raise ValueError(f"row {number}, column job_id: duplicate {job.job_id}")
            seen.add(job.job_id)
            jobs.append(job)
    return jobs

def _trace_rows(path: str | Path, trace_format: str) -> list[dict]:
    records = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        if trace_format == "simple":
            reader = csv.DictReader(stream)
            needed = {"job_submit_time", "num_nodes", "time_limit"}
            missing = needed - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"missing required column {sorted(missing)[0]}")
            for number, row in enumerate(reader, start=2):
                runtime = row.get("actual_run_time")
                records.append({
                    "submit": _parse(row["job_submit_time"], "job_submit_time"),
                    "nodes": _parse(row["num_nodes"], "num_nodes", int),
                    "limit": _parse(row["time_limit"], "time_limit"),
                    "runtime": None if runtime in (None, "") else
                    _parse(runtime, "actual_run_time"),
                    "user": row.get("user"),
                })
        else:
            reader = csv.reader(stream)
            next(reader, None)
            for number, row in enumerate(reader, start=2):
                if len(row) < 33:
                    raise ValueError(
                        f"row {number}: Lassen input needs 33 columns"
                    )
                hint = "; convert the file first"
                begin = _parse(row[22], "23", hint=hint)
                end = _parse(row[23], "24", hint=hint)
                records.append({
                    "submit": _parse(row[28], "29", hint=hint),
                    "nodes": _parse(row[10], "11", int),
                    "limit": _parse(row[31], "32"),
                    "runtime": end - begin,
                    "user": None,
                })
    return records

def _persona_bid(seed: int, user: str, job_id: str) -> tuple[str, float]:
    import numpy
    def generator(key: str):
        digest = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        return numpy.random.default_rng(digest)
    user_rng = generator(f"{seed}:{user}")
    job_rng = generator(f"{seed}:{user}:{job_id}")
    draw, heavy = user_rng.random(), user_rng.random() < 0.2
    if draw < 0.45:
        return "sticker", 1.0
    if draw < 0.80:
        urgent = job_rng.random() < 0.2
        return "tier", 4.0 if urgent and heavy else 2.0 if urgent else 1.0
    if draw < 0.95:
        return "value", round(job_rng.lognormal(math.log(3.0), 0.5), 4)
    return "whale", 10.0

def prepare(
    trace: str | Path, *, trace_format: str = "simple", start=None,
    hours=None, seed: int = 0, requires: str = "",
) -> tuple[list[Job], dict[str, int], list[dict]]:
    """Prepare deterministic market jobs from a contiguous trace interval."""
    if trace_format not in {"simple", "lassen"}:
        raise ValueError("trace_format must be 'simple' or 'lassen'")
    records = _trace_rows(trace, trace_format)
    summary = {"read": len(records), "kept": 0, "no_nodes": 0,
               "no_limit": 0, "bad_runtime": 0, "outside_interval": 0}
    lower = records[0]["submit"] if start is None and records else float(start or 0)
    upper = math.inf if hours is None else lower + float(hours) * 3600
    selected = [r for r in records if lower <= r["submit"] < upper]
    summary["outside_interval"] = len(records) - len(selected)
    selected.sort(key=lambda record: record["submit"])
    kept = []
    for record in selected:
        record["limit"] = int(record["limit"])
        runtime = record["runtime"]
        record["runtime"] = None if runtime is None else int(runtime)
        reason = "no_nodes" if record["nodes"] < 1 else None
        reason = "no_limit" if reason is None and record["limit"] < 1 else reason
        bad_runtime = record["runtime"] is not None and record["runtime"] < 1
        reason = "bad_runtime" if reason is None and bad_runtime else reason
        if reason is None:
            kept.append(record)
        else:
            summary[reason] += 1
    origin = kept[0]["submit"] if kept else 0
    requirement_text = " ".join(requires.split())
    requirement_set = frozenset(requirement_text.split())
    jobs, rows = [], []
    for index, record in enumerate(kept, start=1):
        job_id = f"j{index:06d}"
        user = record["user"] if record["user"] is not None else job_id
        persona, bid = _persona_bid(seed, user, job_id)
        job = Job(job_id, int(record["submit"] - origin), record["nodes"],
                  record["limit"], bid, requirement_set, record["runtime"])
        jobs.append(job)
        rows.append(dict(zip(_JOB_FIELDS, (
            job.job_id, job.submit_s, job.num_nodes, job.limit_s, job.bid,
            requirement_text, "" if job.runtime_s is None else job.runtime_s,
            user, persona,
        ))))
    summary["kept"] = len(jobs)
    return jobs, summary, rows

def write_jobs(rows: list[dict], path: str | Path) -> None:
    """Write prepared job rows with stable columns and Unix line endings."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_JOB_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
