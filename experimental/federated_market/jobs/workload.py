################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""A workload model: fit day types to traces, then generate days of jobs.

An array is jobs of one node count submitted at most a minute apart, as a script sends
them. The fit splits a trace's whole days into thirds by the work they hold, light,
medium and busy days, and keeps each day's arrays, each with its hour, node count,
size, span and run times. A generated day of a type takes one of that type's days as
its outline, and refills each array with a random real array of the same kind: the
same quarter of the day, node class and decade of work. A refill arrives at its own
hour and a new minute, its jobs spread over its real span, with fresh run times. A
profile holds no job identity, no user and no date.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from .traces import Row, read_lc, read_simple

_DAY = 86400
DAY_TYPES = ("light", "medium", "busy")
# An array keeps at most this many quantiles of its jobs' log run times.
_RUN_LEVELS = 9
# Node classes for an array's kind: 1, 2-4, 5-16, 17-64 and wider.
_NODE_CLASSES = (1, 4, 16, 64)


def _run_time(row):
    """Return a usable row's run time, or None when the row cannot be modeled."""
    if row.nodes is None or row.nodes < 1 or not row.ran:
        return None
    run = row.runtime if row.runtime is not None else row.limit
    return run if run >= 1 else None


def _draw(quantiles, u):
    """Return the value at probability u, between the stored quantiles."""
    if len(quantiles) == 1:
        return quantiles[0]
    position = u * (len(quantiles) - 1)
    low = min(int(position), len(quantiles) - 2)
    return quantiles[low] + (position - low) * (quantiles[low + 1] - quantiles[low])


def _mean_run(runs):
    """Return the mean run time, in seconds, that an array's log quantiles give."""
    if len(runs) == 1:
        return math.exp(runs[0])
    total = 0.0
    for a, b in zip(runs, runs[1:]):
        total += math.exp(a) if a == b else (math.exp(b) - math.exp(a)) / (b - a)
    return total / (len(runs) - 1)


def _work(array):
    """Return an array's expected node-hours."""
    _, nodes, size, _, runs = array
    return nodes * size * _mean_run(runs) / 3600


def _kinds(array):
    """Return an array's kind, then looser kinds to refill from when it has no match."""
    width = sum(array[1] > bound for bound in _NODE_CLASSES)
    decade = min(max(math.floor(math.log10(_work(array))), -2), 3)
    band = array[0] // 6
    return (band, width, decade), (band, None, decade), (None, None, decade), ()


def _array(array, offset_s):
    """Return an array's hour, node count, size, span and quantiles of log run time."""
    import numpy

    logs = numpy.log([_run_time(row) for row in array])
    if logs.min() == logs.max():
        runs = [round(float(logs[0]), 3)]
    else:
        levels = numpy.linspace(0.0, 1.0, min(len(logs), _RUN_LEVELS))
        runs = [round(float(value), 3) for value in numpy.quantile(logs, levels)]
    hour = int((array[0].submit + offset_s) % _DAY // 3600)
    return [hour, array[0].nodes, len(array), array[-1].submit - array[0].submit, runs]


def _fit_source(rows, whole, offset_s, array_gap_s):
    import numpy

    by_nodes = defaultdict(list)
    for row in rows:
        by_nodes[row.nodes].append(row)
    arrays, work = defaultdict(list), defaultdict(float)
    for own in by_nodes.values():
        own.sort(key=lambda row: (row.submit, row.order))
        array = [own[0]]
        for row in [*own[1:], None]:
            if row is not None and row.submit - array[-1].submit <= array_gap_s:
                array.append(row)
                continue
            day = math.floor((array[0].submit + offset_s) / _DAY)
            if day in whole:
                arrays[day].append(_array(array, offset_s))
                work[day] += sum(job.nodes * _run_time(job) for job in array)
            array = [row]
    # Days go into thirds by the work they hold, lightest first.
    order = sorted(whole, key=lambda day: (work[day], day))
    types = {}
    for name, days in zip(DAY_TYPES, numpy.array_split(order, len(DAY_TYPES))):
        types[name] = [arrays[day] for day in days]
    return {
        "jobs": len(rows),
        "run_time_max_s": max(_run_time(row) for row in rows),
        "types": types,
    }


def fit_profiles(
    traces,
    *,
    trace_format="lc",
    start=None,
    hours=None,
    utc_offset_hours=0.0,
    array_gap_s=60,
) -> dict:
    """Fit one profile to each named trace, over an optional interval of its clock.

    Hours of the day are read at `utc_offset_hours` from UTC, and only the interval's
    whole days are typed; without an interval, the trace runs from its first submit to
    its last. Jobs of one node count at most `array_gap_s` apart form an array.
    """
    readers = {"lc": read_lc, "simple": read_simple}
    if trace_format not in readers:
        raise ValueError("trace_format must be 'lc' or 'simple'")
    offset_s = round(utc_offset_hours * 3600)
    profiles = {}
    for source, path in traces.items():
        rows = [row for row in readers[trace_format](source, path) if _run_time(row)]
        first = min((row.submit for row in rows), default=0) if start is None else start
        end = max((row.submit + 1 for row in rows), default=0)
        if start is not None and hours is not None:
            end = start + hours * 3600
        rows = [row for row in rows if first <= row.submit < end]
        if not rows:
            raise ValueError(f"{source}: no usable jobs to fit")
        whole = range(
            math.ceil((first + offset_s) / _DAY), math.floor((end + offset_s) / _DAY)
        )
        if len(whole) < len(DAY_TYPES):
            raise ValueError(f"{source}: fewer than three whole days to fit")
        profiles[source] = _fit_source(rows, whole, offset_s, array_gap_s)
    return profiles


def write_profiles(profiles, path) -> None:
    """Write profiles as JSON."""
    Path(path).write_text(json.dumps(profiles, indent=1) + "\n", encoding="utf-8")


def read_profiles(path) -> dict:
    """Read profiles written by write_profiles."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _days_for(days, source):
    """Return a source's days: a count of days of random types, or a list of types."""
    chosen = days.get(source) if isinstance(days, dict) else days
    if isinstance(chosen, int) and not isinstance(chosen, bool) and chosen >= 1:
        return chosen
    if isinstance(chosen, (list, tuple)) and chosen:
        if all(kind in DAY_TYPES for kind in chosen):
            return list(chosen)
    raise ValueError("days must be a positive count, or a list of light, medium, busy")


def _pools(profile, nodes):
    """Return the arrays within the node range by kind, and by each looser kind."""
    low, high = nodes
    pools = defaultdict(list)
    for outlines in profile["types"].values():
        for outline in outlines:
            for array in outline:
                if low <= array[1] and (high is None or array[1] <= high):
                    for kind in _kinds(array):
                        pools[kind].append(array)
    return pools


def _refills(pools, array):
    """Return the arrays that can refill an array: its own kind, or the nearest one."""
    return next((pools[kind] for kind in _kinds(array) if pools.get(kind)), [])


def _source_rows(source, profile, days, seed, volume, nodes, spread):
    import numpy

    from .bids import generator

    # Day types, outlines and copies have their own stream, so a node range keeps them.
    levels = generator(seed, source, "#days")
    rng = generator(seed, source, "#workload")
    if isinstance(days, int):
        days = [DAY_TYPES[i] for i in levels.integers(len(DAY_TYPES), size=days)]
    pools = _pools(profile, nodes)
    rows, arrays = [], 0
    for day, kind in enumerate(days):
        outlines = profile["types"][kind]
        outline = outlines[levels.integers(len(outlines))]
        scale = volume * math.exp(spread * levels.standard_normal() - spread**2 / 2)
        for template in outline:
            copies = int(scale + levels.random())
            pool = _refills(pools, template)
            for _ in range(copies if pool else 0):
                hour, width, size, span, runs = pool[rng.integers(len(pool))]
                start = day * _DAY + 3600 * hour + rng.uniform(0, 3600)
                offsets = [0.0] + list(numpy.sort(rng.uniform(0, span, size - 1)))
                arrays += 1
                for offset in offsets:
                    run = round(math.exp(_draw(runs, rng.random())))
                    order = len(rows)
                    rows.append(
                        Row(
                            source,
                            order,
                            f"generated:{order}",
                            math.floor(start + offset),
                            int(width),
                            None,
                            min(max(1, run), profile["run_time_max_s"]),
                            f"a{arrays:06d}",
                            True,
                        )
                    )
    return rows, {"arrays": arrays, "days": days}


def generated_rows(
    profiles, *, days=1, seed=0, volume=1.0, nodes=(1, None), spread=0.0
):
    """Return the generated trace rows of every source, merged by submit time, and
    each source's arrays and day types.

    `days` is a number of days of random types, a list of types (`light`, `medium`,
    `busy`) for every source, or a dict of either by source. Day d starts at d x 86400 s,
    at midnight of the profiles' hours. `volume` copies every array of an outline that
    many times on average, and `spread` varies each day's volume, log-normally. Refills
    come only from the node range `nodes`, (low, high) with high None for no limit: an
    array outside it is refilled from the range at its time of day and work.
    """
    low, high = nodes
    if low < 1 or (high is not None and high < low):
        raise ValueError("nodes must be a range of at least one node, in order")
    if volume < 0 or spread < 0:
        raise ValueError("volume and spread must not be negative")
    rows, found = [], {}
    for source, profile in profiles.items():
        made, found[source] = _source_rows(
            source, profile, _days_for(days, source), seed, volume, nodes, spread
        )
        rows += made
    rows.sort(key=lambda row: (row.submit, row.source, row.order))
    return rows, found


def daily_node_hours(profiles, *, days=1, nodes=(1, None)) -> float:
    """Return the node-hours of work a generated day holds on average, at volume 1."""
    total = 0.0
    for source, profile in profiles.items():
        chosen = _days_for(days, source)
        kinds = DAY_TYPES if isinstance(chosen, int) else chosen
        pools = _pools(profile, nodes)
        means = {}
        for kind in kinds:
            outlines = profile["types"][kind]
            for outline in outlines:
                for template in outline:
                    pool = _refills(pools, template)
                    if id(pool) not in means:
                        held = sum(_work(array) for array in pool)
                        means[id(pool)] = held / len(pool) if pool else 0.0
                    total += means[id(pool)] / len(outlines) / len(kinds)
    return total
