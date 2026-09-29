################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""From raw traces to a job stream: interval, cleaning, ids, bids."""

import math
from numbers import Real

from .bids import generator, persona_terms, price_bid
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
    reference_price,
    speeds=None,
    trace_format="lc",
    start=None,
    hours=None,
    seed=0,
    gpu_fraction=0.5,
    requires=None,
    per_platform=None,
) -> tuple[list[Job], dict[str, int]]:
    """Prepare deterministic jobs from one interval across named traces."""
    readers = {"lc": read_lc, "simple": read_simple}
    if trace_format not in readers:
        raise ValueError("trace_format must be 'lc' or 'simple'")
    if per_platform is not None and (
        speeds is None or any(name not in speeds for name in per_platform)
    ):
        raise ValueError("per-platform bids need every platform's speeds")
    if (
        isinstance(gpu_fraction, bool)
        or not isinstance(gpu_fraction, Real)
        or not 0 <= gpu_fraction <= 1
    ):
        raise ValueError("gpu_fraction must be a number in [0, 1]")
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
    requirement_override = None if requires is None else frozenset(requires.split())
    jobs = []
    for index, record in enumerate(kept, start=1):
        job_rng = generator(seed, record.source, record.identity)
        persona, persona_price, preference_rng = persona_terms(
            seed, record.source, record.user, reference_price, job_rng
        )
        requirement_set = requirement_override
        if requirement_set is None:
            requirement_set = (
                frozenset({"gpu"}) if job_rng.random() < gpu_fraction else frozenset()
            )
        hardware = "gpu" if "gpu" in requirement_set else "cpu"
        job_speeds = (
            None
            if per_platform is None
            else {
                name: speeds[name][hardware]
                for name in per_platform
                if hardware in speeds[name]
            }
        )
        bid = price_bid(persona_price, per_platform, job_speeds, preference_rng)
        jobs.append(
            Job(
                f"j{index:06d}",
                record.submit - origin,
                record.nodes,
                record.limit if record.runtime is None else record.runtime,
                bid,
                requirement_set,
                record.runtime,
                record.limit,
                record.source,
                record.user or "",
                persona,
            )
        )
        summary[f"kept:{record.source}"] += 1
    summary["kept"] = len(jobs)
    return jobs, summary
