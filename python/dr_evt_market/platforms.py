################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""Platforms: one dr_evt simulation per machine, and the five LLNL machines."""

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

# The market submits only jobs that start at once, so dr_evt's queue statistics are
# empty by construction; these are the ones that say something.
_STATISTIC_FIELDS = ("jobs_completed", "utilization", "makespan")


class Platform:
    """One real machine, exposing a share of itself as a dr_evt simulation."""

    name: str
    total_nodes: int
    price_per_node_hour: float
    hardware: frozenset[str]
    speed: dict[str, float]

    def __init__(self, work_dir: str | Path, share: float = 1.0) -> None:
        """Set the exposed share; the dr_evt simulation starts on first use."""
        if (
            isinstance(share, bool)
            or not isinstance(share, Real)
            or not 0 <= share <= 1
        ):
            raise ValueError("share must be a number in [0, 1]")
        self.share = float(share)
        self.exposed_nodes = round(self.total_nodes * self.share)
        self._work_dir = Path(work_dir).resolve()
        self._simulation = None
        self._last_time_s = 0

    def _started(self):
        # Training reads only a platform's facts, so dr_evt loads when a market runs.
        if self._simulation is None:
            import dr_evt

            self._work_dir.mkdir(parents=True, exist_ok=True)
            header = self._work_dir / f"{self.name}.csv"
            header.write_text(
                "job_submit_time,num_nodes,q_id,time_limit\n", encoding="utf-8"
            )
            params = dr_evt.SimParams()
            params.infile = str(header)
            params.total_nodes = self.exposed_nodes
            params.trace_format = "simple"
            params.timestamp_format = "epoch"
            params.run_time_mode = dr_evt.RunTimeMode.LIMIT
            params.backfill_policy = dr_evt.BackfillPolicy.EASY
            params.priority_policy = dr_evt.PriorityPolicy.FCFS
            # The simulation reads its parameters by reference: keep them alive.
            self._params = params
            self._dr_evt = dr_evt
            self._simulation = dr_evt.Simulation(params)
        return self._simulation

    def fits(self, job) -> bool:
        """Return whether a job's hardware and node demand fit this platform."""
        return job.requires <= self.hardware and job.num_nodes <= self.exposed_nodes

    def job_speed(self, job) -> float:
        """Return this platform's speed for a job's hardware."""
        hardware = "gpu" if "gpu" in job.requires else "cpu"
        return self.speed[hardware]

    def run_time(self, job) -> int:
        """Return the integer time for which this job holds nodes."""
        return max(1, math.ceil(job.limit_s / self.job_speed(job)))

    def cost(self, job) -> float:
        """Return the platform cost of the job's priced reservation."""
        return (
            self.price_per_node_hour
            * job.num_nodes
            * job.limit_s
            / self.job_speed(job)
            / 3600
        )

    def advance_to(self, time_s: int) -> None:
        """Advance the simulation to a nondecreasing integer time."""
        if isinstance(time_s, bool) or not isinstance(time_s, int):
            raise ValueError("advance time must be an integer")
        if time_s < self._last_time_s:
            raise ValueError("advance time cannot move backwards")
        self._started().advance_to(time_s)
        self._last_time_s = time_s

    def free_nodes(self) -> int:
        """Return the number of nodes currently available."""
        return int(self._started().get_available_nodes())

    def waiting(self) -> int:
        """Return the number of jobs waiting for scheduler placement."""
        return int(self._started().get_active_job_count())

    def submit(self, jobs, time_s: int) -> None:
        """Submit jobs at one time, in their given order."""
        simulation = self._started()
        requests = [
            self._dr_evt.JobAppendRequest(
                time_s, job.num_nodes, "1", self.run_time(job)
            )
            for job in jobs
        ]
        if requests:
            simulation.append_jobs(requests)

    def statistics(self) -> dict[str, float]:
        """Return the simulation's completed jobs, utilization and makespan."""
        statistics = self._started().get_statistics()
        return {field: float(getattr(statistics, field)) for field in _STATISTIC_FIELDS}


# Speeds were derived on 2026-09-28 from ranks == 1 rows by removing the
# "(via quartz)" marks, taking each platform column's median relative run time,
# and using its reciprocal. The sample has 635 rows across 8 applications.
# Corona has no matrix row and uses 1.0. Dane's CPU values are bimodal, so its
# median-based speed is sensitive to the aggregation rule.
#
# Node counts: Corona's and Tuolumne's are LLNL's published numbers. Tioga's page
# lists 24 compute nodes; it keeps 32 until its size is confirmed. Posted prices per
# node-hour, like Dane's and Matrix's node counts, were set on 2026-09-28.


class Corona(Platform):
    """The Corona GPU platform profile."""

    name = "corona"
    total_nodes = 121
    price_per_node_hour = 1.5
    hardware = frozenset({"cpu", "gpu"})
    speed = {"cpu": 1.0, "gpu": 1.0}


class Dane(Platform):
    """The Dane CPU platform profile."""

    name = "dane"
    total_nodes = 1544
    price_per_node_hour = 0.18
    hardware = frozenset({"cpu"})
    speed = {"cpu": 0.861}


class Matrix(Platform):
    """The Matrix GPU platform profile."""

    name = "matrix"
    total_nodes = 30
    price_per_node_hour = 1.6
    hardware = frozenset({"cpu", "gpu"})
    speed = {"cpu": 2.574, "gpu": 3.695}


class Tioga(Platform):
    """The Tioga GPU platform profile."""

    name = "tioga"
    total_nodes = 32
    price_per_node_hour = 2.7
    hardware = frozenset({"cpu", "gpu"})
    speed = {"cpu": 1.594, "gpu": 7.042}


class Tuolumne(Platform):
    """The Tuolumne GPU platform profile."""

    name = "tuolumne"
    total_nodes = 1152
    price_per_node_hour = 0.19
    hardware = frozenset({"cpu", "gpu"})
    speed = {"cpu": 1.401, "gpu": 3.313}


PLATFORMS = (Corona, Dane, Matrix, Tioga, Tuolumne)
DEFAULT_FEDERATION = ("corona", "dane", "matrix", "tioga", "tuolumne")
PROFILES = {profile.name: profile for profile in PLATFORMS}


def federation(work_dir, share=1.0, names=DEFAULT_FEDERATION) -> dict[str, Platform]:
    """Build the selected named platforms in the requested order."""
    root = Path(work_dir)
    mapped = isinstance(share, Mapping)
    try:
        return {
            name: PROFILES[name](root / name, share.get(name, 1.0) if mapped else share)
            for name in names
        }
    except KeyError as error:
        raise ValueError(f"unknown platform {error.args[0]!r}") from None
