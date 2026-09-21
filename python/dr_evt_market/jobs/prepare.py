################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""From raw traces to a job stream: interval, cleaning, ids, bids."""

import math

from .bids import persona_bid
from .job import _JOB_FIELDS, Job
from .traces import _read_trace


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
        persona, bid = persona_bid(seed, record["source"], user, job_id, per_platform)
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
