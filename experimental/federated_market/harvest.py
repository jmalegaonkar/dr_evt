################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Harvest market windows for training RegretFormer, and save and load them."""

import gzip
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from .jobs import Job
from .market import run
from .mechanism import Mechanism, Vcg
from .platforms import DEFAULT_FEDERATION, federation


class _Recorder(Mechanism):
    """Record each window with jobs queued while another mechanism decides."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = inner.name
        self.windows = []

    def offers(self, job, platforms, free_nodes):
        """Return the inner mechanism's offers, which the market checks against."""
        return self.inner.offers(job, platforms, free_nodes)

    def decide(self, jobs, platforms, free_nodes):
        """Record the window, then return the inner mechanism's decisions."""
        jobs = list(jobs)
        if jobs:
            self.windows.append((jobs, dict(free_nodes)))
        return self.inner.decide(jobs, platforms, free_nodes)


def record_windows(jobs, platforms, *, mechanism=None, window_s=60, prefix=32):
    """Run the market and return every window's batch and free nodes.

    The queue, and so every window, follows the mechanism that decides: VCG unless
    another is given.
    """
    recorder = _Recorder(Vcg() if mechanism is None else mechanism)
    run(jobs, platforms, recorder, window_s=window_s, prefix=prefix)
    return recorder.windows


def _facts(platforms):
    return {
        name: {
            "total_nodes": platform.total_nodes,
            "exposed_nodes": platform.exposed_nodes,
            "price_per_node_hour": platform.price_per_node_hour,
            "hardware": sorted(platform.hardware),
            "speed": platform.speed,
        }
        for name, platform in platforms.items()
    }


def harvest(
    streams,
    mechanisms,
    *,
    share=1.0,
    names=DEFAULT_FEDERATION,
    window_s=60,
    prefix=32,
):
    """Run every stream under every mechanism and return the windows they auction.

    `streams` maps a label to a list of jobs, and `mechanisms` maps a label to a
    function that builds a fresh mechanism. Each run starts from idle platforms, and
    each window comes back with its stream, its mechanism, its free nodes and its batch.
    """
    windows = []
    for stream, jobs in streams.items():
        for label, make in mechanisms.items():
            with tempfile.TemporaryDirectory() as directory:
                platforms = federation(directory, share, names=names)
                recorded = record_windows(
                    jobs, platforms, mechanism=make(), window_s=window_s, prefix=prefix
                )
            windows += [
                {"stream": stream, "mechanism": label, "free": free, "jobs": batch}
                for batch, free in recorded
            ]
    return windows


def write_windows(path, windows, platforms) -> None:
    """Write windows as gzipped JSON lines: the federation's facts, then each window."""
    with gzip.open(Path(path), "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"platforms": _facts(platforms)}) + "\n")
        for window in windows:
            jobs = [
                {**asdict(job), "requires": sorted(job.requires)}
                for job in window["jobs"]
            ]
            stream.write(json.dumps({**window, "jobs": jobs}) + "\n")


def read_windows(paths, work_dir):
    """Read windows files from one federation; return its platforms and the windows.

    The platforms are rebuilt from the profiles at the recorded shares, without
    starting dr_evt, and must match the facts the windows were harvested with.
    """
    facts, windows = None, []
    for path in paths:
        with gzip.open(Path(path), "rt", encoding="utf-8") as stream:
            header = json.loads(next(stream))["platforms"]
            if facts is None:
                facts = header
            elif header != facts:
                raise ValueError(f"{path}: harvested on another federation")
            for line in stream:
                window = json.loads(line)
                window["jobs"] = [
                    Job(**{**job, "requires": frozenset(job["requires"])})
                    for job in window["jobs"]
                ]
                windows.append(window)
    if facts is None:
        raise ValueError("no windows files")
    shares = {
        name: (
            fact["exposed_nodes"] / fact["total_nodes"] if fact["total_nodes"] else 0.0
        )
        for name, fact in facts.items()
    }
    platforms = federation(work_dir, shares, names=list(facts))
    if _facts(platforms) != facts:
        raise ValueError("the windows were harvested with other platform profiles")
    return platforms, windows
