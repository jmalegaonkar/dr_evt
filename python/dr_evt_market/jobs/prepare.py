################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""From raw traces to a job stream: interval, cleaning, ids, bids."""

import math
from numbers import Real

from .bids import PERSONAS, generator, multi_bid, single_bid, terms
from .job import Job
from .synthetic import synthesize
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
    platforms,
    bids="multi",
    trace_format="lc",
    start=None,
    hours=None,
    seed=0,
    gpu_fraction=0.5,
    requires=None,
    synthetic=False,
) -> tuple[list[Job], dict[str, int]]:
    """Prepare deterministic jobs from one interval across named traces.

    `platforms` are the profiles bid on, each with a `name`, a posted
    `price_per_node_hour`, its `hardware` and its `speed`. `bids` is "multi", a bid
    on each platform, or "single", one bid for all. With `synthetic`, the jobs are one
    day drawn from the interval instead.
    """
    readers = {"lc": read_lc, "simple": read_simple}
    if trace_format not in readers:
        raise ValueError("trace_format must be 'lc' or 'simple'")
    if bids not in {"single", "multi"}:
        raise ValueError("bids must be 'single' or 'multi'")
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
            summary[f"kept:{record.source}"] += 1
        else:
            summary[reason] += 1
    summary["kept"] = len(kept)
    if synthetic:
        last = max((row.submit for row in selected), default=lower)
        end = upper if upper < math.inf else last + 1
        kept, found = synthesize(kept, lower, end, seed)
        for source in traces:
            summary[f"groups:{source}"] = found.get(source, 0)
            summary[f"synthetic:{source}"] = sum(row.source == source for row in kept)
    summary.update({f"persona:{name}": 0 for name in PERSONAS})

    origin = kept[0].submit if kept else 0
    requirement_override = None if requires is None else frozenset(requires.split())
    jobs = []
    for index, record in enumerate(kept, start=1):
        job_rng = generator(seed, record.source, record.identity)
        # The hardware draw comes first and is always made, so neither an override
        # nor the bid rule moves any other draw.
        gpu = job_rng.random() < gpu_fraction
        requirement_set = requirement_override
        if requirement_set is None:
            requirement_set = frozenset({"gpu"}) if gpu else frozenset()
        job_terms = terms(seed, record.source, record.user, job_rng)
        usable = [item for item in platforms if requirement_set <= item.hardware]
        if not usable:
            bid = {}
        elif bids == "multi":
            owner = record.identity if record.user is None else record.user
            prices = {item.name: item.price_per_node_hour for item in usable}
            bid = multi_bid(seed, record.source, owner, job_terms, prices)
        else:
            # The price level of a unit of work, averaged over the usable platforms.
            hardware = "gpu" if "gpu" in requirement_set else "cpu"
            level = sum(
                item.price_per_node_hour / item.speed[hardware] for item in usable
            ) / len(usable)
            bid = single_bid(job_terms, level)
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
                job_terms.persona,
            )
        )
        summary[f"persona:{job_terms.persona}"] += 1
    return jobs, summary
