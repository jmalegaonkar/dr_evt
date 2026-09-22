################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""From raw traces to a job stream: interval, cleaning, ids, bids."""

import math

from .bids import persona_bid
from .job import Job
from .traces import read_lc, read_simple


def _drop_reason(row):
    if row.nodes is None or row.nodes < 1:
        return "no_nodes"
    if row.limit < 1:
        return "no_limit"
    if not row.ran:
        return "no_start"
    if row.runtime is not None and row.runtime < 1:
        return "bad_runtime"
    return None


def prepare(
    traces,
    *,
    home_prices,
    trace_format="lc",
    start=None,
    hours=None,
    seed=0,
    requires="gpu",
    per_platform=None,
    limit_from="runtime",
) -> tuple[list[Job], dict[str, int]]:
    """Prepare deterministic jobs from one interval across named traces."""
    readers = {"lc": read_lc, "simple": read_simple}
    if trace_format not in readers:
        raise ValueError("trace_format must be 'lc' or 'simple'")
    if any(source not in home_prices for source in traces):
        raise ValueError("every trace source must have a home price")
    if limit_from not in {"runtime", "request"}:
        raise ValueError("limit_from must be 'runtime' or 'request'")
    reader = readers[trace_format]
    records = []
    for source, path in traces.items():
        records.extend(reader(source, path))
    records.sort(key=lambda row: (row.submit, row.source, row.order))

    lower = (
        records[0].submit
        if start is None and records
        else math.floor(float(start or 0))
    )
    upper = math.inf if hours is None else lower + float(hours) * 3600
    selected = [row for row in records if lower <= row.submit < upper]
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
        reason = _drop_reason(record)
        if reason is None:
            kept.append(record)
            continue
        summary[reason] += 1

    origin = kept[0].submit if kept else 0
    requirement_set = frozenset(requires.split())
    jobs = []
    for index, record in enumerate(kept, start=1):
        job_id = f"j{index:06d}"
        user = record.user if record.user is not None else job_id
        persona, bid = persona_bid(
            seed,
            record.source,
            user,
            job_id,
            home_prices[record.source],
            per_platform,
        )
        limit = (
            record.runtime
            if limit_from == "runtime" and record.runtime is not None
            else record.limit
        )
        jobs.append(
            Job(
                job_id,
                record.submit - origin,
                record.nodes,
                limit,
                bid,
                requirement_set,
                record.runtime,
                record.limit,
                record.source,
                user,
                persona,
            )
        )
        summary[f"kept:{record.source}"] += 1
    summary["kept"] = len(jobs)
    return jobs, summary
