################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""A workload model: fit day types to traces, then generate days of jobs.

The traces' jobs are merged into one stream, as the market sees them, over the days that
every trace covers in whole. The fit splits those days into thirds by the work they
hold, light, medium and busy, and keeps each day's stream: every job's source, second
of the day, node count and run time. A generated day of a type replays one of that
type's streams. Every job keeps its source and its second, except that bursts, one
source's jobs at most a minute apart, shift together by a random offset; and every job
is refilled by a random real job of the same source and kind, from any day: the same
quarter of the day, node class and decade of work. A profile holds no job identity, no
user and no date.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from .traces import Row, read_lc, read_simple

_DAY = 86400
DAY_TYPES = ("light", "medium", "busy")
# One source's jobs at most this far apart form a burst, which shifts as one.
_BURST_GAP_S = 60
# Node classes for a job's kind: 1, 2-4, 5-16, 17-64 and wider.
_NODE_CLASSES = (1, 4, 16, 64)


def _run_time(row):
    """Return a usable row's run time, or None when the row cannot be modeled."""
    if row.nodes is None or row.nodes < 1 or not row.ran:
        return None
    run = row.runtime if row.runtime is not None else row.limit
    return run if run >= 1 else None


def _kinds(job):
    """Return a job's kind, then looser kinds to refill from when it has no match."""
    source, second, nodes, run = job
    width = sum(nodes > bound for bound in _NODE_CLASSES)
    decade = min(max(math.floor(math.log10(nodes * run / 3600)), -2), 3)
    quarter = second // (_DAY // 4)
    return (
        (source, quarter, width, decade),
        (source, quarter, None, decade),
        (source, None, None, decade),
        (source, None, None, None),
        (),
    )


def fit_profile(
    traces, *, trace_format="lc", start=None, hours=None, utc_offset_hours=0.0
) -> dict:
    """Fit one profile to the named traces, merged into one stream, over an optional
    interval of their clock.

    Hours of the day are read at `utc_offset_hours` from UTC, and only days that every
    trace covers in whole are typed; without an interval, a trace runs from its first
    submit to its last.
    """
    readers = {"lc": read_lc, "simple": read_simple}
    if trace_format not in readers:
        raise ValueError("trace_format must be 'lc' or 'simple'")
    offset_s = round(utc_offset_hours * 3600)
    jobs, whole = [], None
    for index, (source, path) in enumerate(traces.items()):
        rows = [row for row in readers[trace_format](source, path) if _run_time(row)]
        first = min((row.submit for row in rows), default=0) if start is None else start
        end = max((row.submit + 1 for row in rows), default=0)
        if start is not None and hours is not None:
            end = start + hours * 3600
        rows = [row for row in rows if first <= row.submit < end]
        if not rows:
            raise ValueError(f"{source}: no usable jobs to fit")
        days = range(
            math.ceil((first + offset_s) / _DAY), math.floor((end + offset_s) / _DAY)
        )
        whole = set(days) if whole is None else whole & set(days)
        jobs += [(index, row.submit + offset_s, row) for row in rows]
    if len(whole) < len(DAY_TYPES):
        raise ValueError("fewer than three whole days that every trace covers")
    streams, work = defaultdict(list), defaultdict(float)
    for index, local, row in jobs:
        day = math.floor(local / _DAY)
        if day in whole:
            run = _run_time(row)
            streams[day].append([index, int(local - day * _DAY), row.nodes, run])
            work[day] += row.nodes * run
    # Days go into thirds by the work they hold, lightest first; the lighter thirds
    # take the days left over.
    order = sorted(whole, key=lambda day: (work[day], day))
    size, extra = divmod(len(order), len(DAY_TYPES))
    types, taken = {}, 0
    for n, name in enumerate(DAY_TYPES):
        count = size + (n < extra)
        types[name] = [
            sorted(streams[day], key=lambda job: (job[1], job[0]))
            for day in order[taken : taken + count]
        ]
        taken += count
    return {
        "sources": list(traces),
        "jobs": sum(len(stream) for stream in streams.values()),
        "types": types,
    }


def write_profile(profile, path) -> None:
    """Write a profile as JSON."""
    Path(path).write_text(json.dumps(profile, indent=1) + "\n", encoding="utf-8")


def read_profile(path) -> dict:
    """Read a profile written by write_profile."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _days_for(days):
    """Return days: a count of days of random types, or a list of types."""
    if isinstance(days, int) and not isinstance(days, bool) and days >= 1:
        return days
    if isinstance(days, (list, tuple)) and days:
        if all(kind in DAY_TYPES for kind in days):
            return list(days)
    raise ValueError("days must be a positive count, or a list of light, medium, busy")


def _pools(profile, nodes):
    """Return the jobs within the node range by kind, and by each looser kind."""
    low, high = nodes
    pools = defaultdict(list)
    for streams in profile["types"].values():
        for stream in streams:
            for job in stream:
                if low <= job[2] and (high is None or job[2] <= high):
                    for kind in _kinds(job):
                        pools[kind].append(job)
    return pools


def _refills(pools, job):
    """Return the jobs that can refill a job: its own kind, or the nearest one."""
    return next((pools[kind] for kind in _kinds(job) if pools.get(kind)), [])


def _bursts(stream):
    """Split a time-ordered stream into bursts: one source's jobs a minute apart."""
    last, open_, bursts = {}, {}, []
    for job in stream:
        source, second = job[0], job[1]
        if source in last and second - last[source] <= _BURST_GAP_S:
            open_[source].append(job)
        else:
            open_[source] = [job]
            bursts.append(open_[source])
        last[source] = second
    return bursts


def generated_rows(
    profile, *, days=1, seed=0, volume=1.0, nodes=(1, None), spread=0.0, jitter=600
):
    """Return the generated trace rows, in submit order, and the day types drawn.

    `days` is a number of days of random types, or a list of types: `light`, `medium`,
    `busy`. Day d starts at d x 86400 s, at midnight of the profile's hours. `volume`
    overlays that many streams of the type a day, the last one thinned to its fraction;
    `spread` varies each day's volume, log-normally; and each burst shifts by up to
    `jitter` seconds either way. Refills come only from the node range `nodes`, (low,
    high) with high None for no limit: a job outside it is stood in for by one inside of
    its source, quarter of the day and work, at its own second.
    """
    from .bids import generator

    days = _days_for(days)
    low, high = nodes
    if low < 1 or (high is not None and high < low):
        raise ValueError("nodes must be a range of at least one node, in order")
    if volume < 0 or spread < 0 or jitter < 0:
        raise ValueError("volume, spread and jitter must not be negative")
    # Day types, streams and times have their own generator, so a node range keeps them.
    levels = generator(seed, "workload", "#days")
    rng = generator(seed, "workload", "#jobs")
    if isinstance(days, int):
        days = [DAY_TYPES[i] for i in levels.integers(len(DAY_TYPES), size=days)]
    pools = _pools(profile, nodes)
    sources = profile["sources"]
    rows, bursts = [], 0
    for day, kind in enumerate(days):
        streams = profile["types"][kind]
        level = volume * math.exp(spread * levels.standard_normal() - spread**2 / 2)
        copies = math.floor(level)
        for copy in range(copies + 1):
            keep = 1.0 if copy < copies else level - copies
            if keep <= 0:
                break
            stream = streams[levels.integers(len(streams))]
            for burst in _bursts(stream):
                shift = levels.uniform(-jitter, jitter)
                taken = [job for job in burst if keep == 1.0 or levels.random() < keep]
                bursts += 1
                for job in taken:
                    pool = _refills(pools, job)
                    if not pool:
                        continue
                    _, _, width, run = pool[rng.integers(len(pool))]
                    order = len(rows)
                    rows.append(
                        Row(
                            sources[job[0]],
                            order,
                            f"generated:{order}",
                            max(0, math.floor(day * _DAY + job[1] + shift)),
                            int(width),
                            None,
                            int(run),
                            f"b{bursts:06d}",
                            True,
                        )
                    )
    rows.sort(key=lambda row: (row.submit, row.source, row.order))
    return rows, days


def daily_node_hours(profile, *, days=1, nodes=(1, None)) -> float:
    """Return the node-hours of work a generated day holds on average, at volume 1."""
    chosen = _days_for(days)
    kinds = DAY_TYPES if isinstance(chosen, int) else chosen
    pools = _pools(profile, nodes)
    means, total = {}, 0.0
    for kind in kinds:
        streams = profile["types"][kind]
        for stream in streams:
            for job in stream:
                pool = _refills(pools, job)
                if id(pool) not in means:
                    held = sum(width * run for _, _, width, run in pool)
                    means[id(pool)] = held / len(pool) / 3600 if pool else 0.0
                total += means[id(pool)] / len(streams) / len(kinds)
    return total
