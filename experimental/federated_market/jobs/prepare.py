################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""From raw traces or a workload profile to a job stream: interval, cleaning, ids, bids."""

import math
from numbers import Real

from .bids import PERSONAS, generator, multi_bid, single_bid, terms
from .job import Job
from .traces import read_lc, read_simple
from .workload import DAY_TYPES, generated_rows


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


def _bid(record, job_terms, usable, hardware, bids, seed):
    """Return a job's bid on the platforms that can run it, in the requested form."""
    if not usable:
        return {}
    if bids == "multi":
        owner = record.identity if record.user is None else record.user
        prices = {item.name: item.price_per_node_hour for item in usable}
        return multi_bid(seed, record.source, owner, job_terms, prices)
    # A single bid prices a unit of work at its mean cost over those platforms.
    level = sum(item.price_per_node_hour / item.speed[hardware] for item in usable)
    return single_bid(job_terms, level / len(usable))


def _check(bids, gpu_fraction):
    if bids not in {"single", "multi"}:
        raise ValueError("bids must be 'single' or 'multi'")
    if (
        isinstance(gpu_fraction, bool)
        or not isinstance(gpu_fraction, Real)
        or not 0 <= gpu_fraction <= 1
    ):
        raise ValueError("gpu_fraction must be a number in [0, 1]")


def _jobs(records, origin, platforms, bids, seed, gpu_fraction, requires, summary):
    """Number the records and give each its hardware and bid; count the personas."""
    summary.update({f"persona:{name}": 0 for name in PERSONAS})
    override = None if requires is None else frozenset(requires.split())
    jobs = []
    for index, record in enumerate(records, start=1):
        job_rng = generator(seed, record.source, record.identity)
        # The hardware draw comes first and is always made, so neither an override
        # nor the bid rule moves any other draw.
        gpu = job_rng.random() < gpu_fraction
        needs = override if override is not None else frozenset({"gpu"} if gpu else ())
        job_terms = terms(seed, record.source, record.user, job_rng)
        usable = [item for item in platforms if needs <= item.hardware]
        hardware = "gpu" if "gpu" in needs else "cpu"
        jobs.append(
            Job(
                f"j{index:06d}",
                record.submit - origin,
                record.nodes,
                record.limit if record.runtime is None else record.runtime,
                _bid(record, job_terms, usable, hardware, bids, seed),
                needs,
                record.runtime,
                record.limit,
                record.source,
                record.user or "",
                job_terms.persona,
            )
        )
        summary[f"persona:{job_terms.persona}"] += 1
    return jobs


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
) -> tuple[list[Job], dict[str, int]]:
    """Prepare deterministic jobs from one interval across named traces.

    `platforms` are the profiles bid on, each with a `name`, a posted
    `price_per_node_hour`, its `hardware` and its `speed`. `bids` is "multi", a bid
    on each platform, or "single", one bid for all.
    """
    readers = {"lc": read_lc, "simple": read_simple}
    if trace_format not in readers:
        raise ValueError("trace_format must be 'lc' or 'simple'")
    _check(bids, gpu_fraction)
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
    origin = kept[0].submit if kept else 0
    jobs = _jobs(kept, origin, platforms, bids, seed, gpu_fraction, requires, summary)
    return jobs, summary


def generate(
    profile,
    *,
    platforms,
    days=1,
    seed=0,
    volume=1.0,
    nodes=(1, None),
    spread=0.0,
    jitter=600,
    bids="multi",
    gpu_fraction=0.5,
    requires=None,
) -> tuple[list[Job], dict[str, int]]:
    """Generate deterministic jobs from a workload profile.

    Day types, streams and refills follow `generated_rows`; hardware and bids follow the
    same rules as `prepare`. Time 0 is midnight of the first day.
    """
    _check(bids, gpu_fraction)
    rows, drawn = generated_rows(
        profile,
        days=days,
        seed=seed,
        volume=volume,
        nodes=nodes,
        spread=spread,
        jitter=jitter,
    )
    summary = {"jobs": len(rows)}
    for source in profile["sources"]:
        summary[f"jobs:{source}"] = sum(row.source == source for row in rows)
    for kind in DAY_TYPES:
        summary[f"days:{kind}"] = drawn.count(kind)
    jobs = _jobs(rows, 0, platforms, bids, seed, gpu_fraction, requires, summary)
    return jobs, summary
